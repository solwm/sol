//! User config loader for sol.
//!
//! Looks for, in order:
//!   1. `$XDG_CONFIG_HOME/sol/sol.conf`
//!   2. `$HOME/.config/sol/sol.conf`
//!   3. built-in defaults — Alt+Return launches alacritty; Alt+D
//!      launches rofi, which is what the compositor had hardcoded
//!      before the config system existed.
//!
//! Missing file is not an error; malformed lines are logged and
//! skipped so a single typo doesn't wipe out the rest of the file.
//!
//! Ctrl+Alt+F1..F12 VT switching stays hardcoded in `apply_input`
//! and is intentionally *not* overridable here — a misconfigured
//! .conf must not be able to trap the user on the compositor.
//!
//! Format (Hyprland-ish):
//!
//! ```text
//! # comments start with #
//! bind         = MOD[+MOD2], KEY, exec, program with args
//! remap        = FROM, TO                  # rewrite a scancode pre-everything
//! exec-once    = program with args         # launched once at startup
//! gaps_in      = 12                        # gap between adjacent tiles, px
//! gaps_out     = 20                        # gap from tile to screen edge, px
//! border_width = 2                         # focused-tile border, px (0 = off)
//! border_color = ffff00                    # hex RRGGBB or #RRGGBB or 0xRRGGBB
//! idle_timeout = 300                       # DPMS-off after N idle seconds (0 = off)
//! mode         = 3840x2160@240             # output WIDTHxHEIGHT@HZ; SOL_MODE env wins
//! keyboard_repeat_rate  = 100              # wl_keyboard.repeat_info chars/sec (0 disables)
//! keyboard_repeat_delay = 200              # wl_keyboard.repeat_info delay, ms
//! animation_duration_ms = 150              # layout-tween duration, 0 = snap
//! animation_curve       = cubic_out        # linear|cubic_out|quart_out|quint_out|expo_out|in_out_cubic
//! workspace_animation   = crossfade        # none|crossfade
//! workspace_animation_duration_ms = 250    # crossfade duration, separate from animation_duration_ms
//! inactive_alpha        = 0.85             # alpha for non-focused toplevels (1.0 = no effect)
//! inactive_blur         = on               # frosted-glass blur behind inactive toplevels
//! inactive_blur_passes  = 4                # blur intensity (more passes = blurrier, costs a bit more GPU)
//! inactive_blur_radius  = 1.0              # per-pass kernel reach in texels (1.0 = default 2-texel reach, 2.0 = 4-texel reach)
//! ```
//!
//! The file is watched at runtime (inotify on its parent dir): saves
//! re-apply gaps, border, idle_timeout, bindings, and remaps live.
//! `mode` changes log a "restart to apply" warning (live mode-set
//! requires GBM/EGL surface rebuild, not yet wired). `exec-once` is
//! startup-only and is ignored on reload.
//!
//! Modifiers (case-insensitive): `ALT`/`MOD1`, `CTRL`/`CONTROL`,
//! `SHIFT`, `SUPER`/`META`/`MOD4`. Keys: letters `A`-`Z`, digits
//! `0`-`9`, `F1`-`F12`, `Return`/`Enter`, `Escape`, `Tab`, `Space`,
//! `Backspace`, `CapsLock`, `Delete`, `Home`, `End`, `PageUp`,
//! `PageDown`, `Insert`, `Left`, `Right`, `Up`, `Down`. Actions:
//! `exec PROGRAM ARGS...` spawns a child process; `focus_dir
//! left|right|up|down` moves keyboard focus to the nearest mapped
//! toplevel in the given direction; `move_dir ...` swaps the
//! focused tile with its neighbor; `toggle_zoom` expands the
//! focused tile to the full usable area (outer gaps respected)
//! or restores the master-stack layout; `close_window` asks the
//! focused toplevel to close via `xdg_toplevel.close`;
//! `workspace N` switches to workspace N (1-based); `move_to_workspace
//! N` sends the focused window to workspace N without following it.
//!
//! `remap` rewrites at the evdev scancode layer, so it's invisible to
//! both bindings and clients: the remapped code is what xkb sees and
//! what gets fed to `wl_keyboard.key`. Built-in default remaps
//! CapsLock to Escape because that's what the primary user expects.
//!
//! `exec-once` commands are spawned at compositor startup, after the
//! Wayland socket is bound and `WAYLAND_DISPLAY` is exported — so a
//! `swaybg` / `hyprpaper` / `mpvpaper` listed here can connect and
//! paint a wallpaper without the user running anything by hand.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};

pub const MOD_ALT: u8 = 1 << 0;
pub const MOD_CTRL: u8 = 1 << 1;
pub const MOD_SHIFT: u8 = 1 << 2;
pub const MOD_SUPER: u8 = 1 << 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModePref {
    pub width: u16,
    pub height: u16,
    pub refresh_hz: u32,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub bindings: Vec<Binding>,
    pub remaps: Vec<Remap>,
    /// Commands spawned once at compositor startup, after the Wayland
    /// socket is ready. Intended for wallpaper daemons, status bars,
    /// notification daemons — any `layer-shell` client the user wants
    /// up without a manual launch.
    pub exec_once: Vec<ExecCommand>,
    pub gaps_in: i32,
    pub gaps_out: i32,
    /// Pixel width of the border drawn around the keyboard-focused
    /// toplevel. 0 disables the border entirely.
    pub border_width: i32,
    /// Straight RGBA in [0, 1]. Parsed from hex RGB in the config file;
    /// alpha is fixed at 1.0 since a translucent border isn't a thing
    /// anyone has asked for yet.
    pub border_color: [f32; 4],
    /// Seconds of no user input before sol turns the monitor off
    /// via DPMS. 0 disables the feature entirely. Clients that want
    /// to prevent this (video players, presentation tools) create
    /// `zwp_idle_inhibitor_v1` inhibitors on their surfaces and the
    /// timer is suppressed while any inhibitor is live.
    pub idle_timeout: u32,
    /// Output mode preference (`mode = WIDTHxHEIGHT@HZ`). When set,
    /// the DRM backend picks this mode at startup (overriding the
    /// preferred-size-max-Hz heuristic). The `SOL_MODE` env var
    /// still takes precedence — env is for one-off testing, config
    /// is the persistent intent. Hot-reloading this value at runtime
    /// is currently a warn-and-keep-old-mode (changing mode requires
    /// rebuilding the GBM/EGL surface; restart sol to apply).
    pub mode: Option<ModePref>,
    /// `wl_keyboard.repeat_info` rate, in characters per second. The
    /// compositor does not implement repeat itself — clients do, off
    /// the rate/delay we advertise. Rate `0` disables auto-repeat.
    /// Default 100 keys/sec to match Hyprland's tuning.
    pub keyboard_repeat_rate: i32,
    /// `wl_keyboard.repeat_info` delay before repeat starts, in
    /// milliseconds. Default 200 ms (Hyprland's tuning).
    pub keyboard_repeat_delay: i32,
    /// Layout-transition animation duration, in milliseconds. 0 means
    /// "snap, no tween." Default 150 ms — long enough that the eye
    /// reads the resize as motion, short enough that keyboard-driven
    /// users don't notice the latency.
    pub animation_duration_ms: u32,
    /// Easing curve applied to layout-transition tweens.
    pub animation_curve: AnimationCurve,
    /// Visual transition for `workspace, N` switches. Default
    /// `Crossfade` — the leaving workspace fades out while the
    /// arriving one fades in. `None` is an instant cut.
    pub workspace_animation: WorkspaceAnimation,
    /// Duration of the workspace transition (`workspace_animation`),
    /// in milliseconds. Independent of `animation_duration_ms` —
    /// workspace switches usually feel right a touch slower than
    /// layout tweens because a workspace switch is a bigger context
    /// shift than a tile resize. Default 250 ms. The shared
    /// `animation_curve` still applies. `0` falls back to an
    /// instant cut just like `workspace_animation = none`.
    pub workspace_animation_duration_ms: u32,
    /// Alpha multiplier applied to any tiled toplevel that doesn't
    /// own keyboard focus. `1.0` disables the effect (inactive looks
    /// identical to active); `0.85` is a subtle tint; lower values
    /// reveal more of the wallpaper / blurred backdrop behind. The
    /// active window always renders at full opacity.
    pub inactive_alpha: f32,
    /// When `true`, the wallpaper / lower layers are blurred under
    /// each inactive toplevel before the toplevel is drawn on top
    /// at `inactive_alpha`. Frosted-glass effect. When `false` the
    /// inactive window's transparency just reveals the un-blurred
    /// wallpaper. Has no visible effect if `inactive_alpha == 1.0`
    /// (window covers the backdrop entirely either way).
    pub inactive_blur: bool,
    /// Number of dual-Kawase blur passes applied to the inactive
    /// backdrop. More passes = blurrier and more expensive. 4 is a
    /// good default for "frosted glass"; 1 is barely-blurred; 8+
    /// is heavily blurred. Each pass is a downsample-then-upsample
    /// pair on a half-resolution FBO, so the cost scales mildly.
    pub inactive_blur_passes: u32,
    /// Per-pass kernel radius scale, in texels. The blur shader
    /// samples a 5×5 box (offsets `dx, dy ∈ {-2..2}`) multiplied
    /// by `u_texel * radius`. `1.0` = the original 2-texel reach
    /// per pass; `2.0` = same kernel but 4-texel reach (sparser
    /// sampling, blurrier-but-noisier). Each pass widens the
    /// effective kernel by convolution, so most users get the
    /// strongest payoff from raising `inactive_blur_passes` first
    /// and only nudging this when they want something dramatic.
    pub inactive_blur_radius: f32,
}

/// Easing functions exposed to the config. All map `t ∈ [0, 1]`
/// (linear progress) to a smoothed output in `[0, 1]`. Names follow
/// the easings.net taxonomy.
///
/// `CubicOut` (the default) front-loads motion and slows to a gentle
/// settle, which reads as "thing snapping into place." `QuartOut` /
/// `QuintOut` are increasingly aggressive front-loads. `ExpoOut`
/// is the most aggressive — almost instant with a tail. `InOutCubic`
/// is symmetric (slow start, slow stop, fast middle), feels more
/// deliberate. `Linear` is no easing — for debugging or taste.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnimationCurve {
    Linear,
    CubicOut,
    QuartOut,
    QuintOut,
    ExpoOut,
    InOutCubic,
}

/// Workspace-switch transition kind. `Crossfade` renders both the
/// outgoing and incoming workspaces together for the duration of
/// the animation, lerping per-window alpha via `animation_curve`.
/// `None` skips the transition and snaps to the new workspace on
/// the same frame as the keybind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceAnimation {
    None,
    Crossfade,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bindings: Vec::new(),
            remaps: Vec::new(),
            exec_once: Vec::new(),
            gaps_in: 12,
            gaps_out: 20,
            border_width: 2,
            border_color: hex_to_rgba(0xFFFF00),
            idle_timeout: 0,
            mode: None,
            keyboard_repeat_rate: 100,
            keyboard_repeat_delay: 200,
            animation_duration_ms: 150,
            animation_curve: AnimationCurve::CubicOut,
            workspace_animation: WorkspaceAnimation::Crossfade,
            workspace_animation_duration_ms: 250,
            inactive_alpha: 0.85,
            inactive_blur: true,
            inactive_blur_passes: 4,
            inactive_blur_radius: 1.0,
        }
    }
}

impl Config {
    /// Apply user-declared remaps to an incoming evdev scancode.
    /// Linear scan — N is tiny and this runs once per keystroke; not
    /// worth a HashMap.
    pub fn remap(&self, code: u32) -> u32 {
        for r in &self.remaps {
            if r.from == code {
                return r.to;
            }
        }
        code
    }
}

#[derive(Debug, Clone)]
pub struct Binding {
    pub mods: u8,
    /// evdev scancode — same domain libinput hands us in `key.key()`.
    pub key: u32,
    pub action: Action,
}

#[derive(Debug, Clone)]
pub struct Remap {
    pub from: u32,
    pub to: u32,
}

#[derive(Debug, Clone)]
pub enum Action {
    Spawn(ExecCommand),
    /// Move keyboard focus to the nearest mapped toplevel in the
    /// given direction (vim-style h/j/k/l navigation).
    FocusDir(Direction),
    /// Swap the currently focused tile with its neighbor in the
    /// given direction — i.e. move the window through the tiling,
    /// keeping focus on it.
    MoveDir(Direction),
    /// Toggle "zoom" on the focused tile: it expands to the full
    /// usable area (outer gaps respected) and every other tile
    /// stops rendering until zoom is toggled off. Not the Wayland
    /// `xdg_toplevel.set_fullscreen` state — we don't tell the
    /// client anything protocol-level; it just receives a larger
    /// configure.
    ToggleZoom,
    /// Ask the focused toplevel to close via `xdg_toplevel.close`.
    /// The client decides what to do (terminals exit, text editors
    /// may prompt to save, etc.). If a client ignores the request
    /// there's no escalation — we're not a window killer.
    CloseWindow,
    /// Switch to the numbered workspace. Workspaces are purely a
    /// compositor concept — clients don't know about them. Toplevels
    /// on the active workspace render and receive input; others stay
    /// mapped but are skipped by the scene walker.
    Workspace(u32),
    /// Move the focused toplevel to the numbered workspace. Focus
    /// falls back to the topmost remaining window on the current
    /// workspace; the moved window keeps whatever rect it had until
    /// the target workspace becomes active and re-lays it out.
    MoveToWorkspace(u32),
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug, Clone)]
pub struct ExecCommand {
    pub program: String,
    pub args: Vec<String>,
}

fn hex_to_rgba(hex: u32) -> [f32; 4] {
    let r = ((hex >> 16) & 0xFF) as f32 / 255.0;
    let g = ((hex >> 8) & 0xFF) as f32 / 255.0;
    let b = (hex & 0xFF) as f32 / 255.0;
    [r, g, b, 1.0]
}

pub fn load() -> Config {
    let path = config_path();
    match std::fs::read_to_string(&path) {
        Ok(text) => {
            let cfg = parse(&text);
            tracing::info!(
                path = %path.display(),
                bindings = cfg.bindings.len(),
                "config loaded"
            );
            cfg
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::info!(
                path = %path.display(),
                "no config file; using built-in defaults"
            );
            default_config()
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "config read failed; using built-in defaults"
            );
            default_config()
        }
    }
}

pub fn config_path() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("sol").join("sol.conf");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config/sol/sol.conf");
    }
    PathBuf::from("sol.conf")
}

pub fn default_config() -> Config {
    let bind_focus = |k: &str, d: Direction| Binding {
        mods: MOD_ALT,
        key: key_from_name(k).unwrap(),
        action: Action::FocusDir(d),
    };
    let bind_move = |k: &str, d: Direction| Binding {
        mods: MOD_ALT | MOD_CTRL,
        key: key_from_name(k).unwrap(),
        action: Action::MoveDir(d),
    };
    Config {
        bindings: vec![
            Binding {
                mods: MOD_ALT,
                key: key_from_name("Return").unwrap(),
                action: Action::Spawn(ExecCommand {
                    program: "alacritty".to_string(),
                    // `-e zsh` so we get the user's real shell with
                    // their ~/.zshrc applied — otherwise alacritty
                    // defaults to $SHELL which on some setups skips
                    // interactive rc files.
                    args: vec!["-e".to_string(), "zsh".to_string()],
                }),
            },
            Binding {
                mods: MOD_ALT,
                key: key_from_name("D").unwrap(),
                action: Action::Spawn(ExecCommand {
                    program: "rofi".to_string(),
                    args: vec!["-show".to_string(), "drun".to_string()],
                }),
            },
            // vim-style focus movement across tiles.
            bind_focus("H", Direction::Left),
            bind_focus("J", Direction::Down),
            bind_focus("K", Direction::Up),
            bind_focus("L", Direction::Right),
            // same layout, Ctrl added: move the focused tile itself.
            bind_move("H", Direction::Left),
            bind_move("J", Direction::Down),
            bind_move("K", Direction::Up),
            bind_move("L", Direction::Right),
            // Alt+Tab zooms the focused tile to fill the screen
            // (still respecting outer gaps). Pressing it again
            // restores the master-stack layout.
            Binding {
                mods: MOD_ALT,
                key: key_from_name("Tab").unwrap(),
                action: Action::ToggleZoom,
            },
            // Alt+Q closes the focused window (polite request via
            // xdg_toplevel.close; the client has final say). Alt
            // rather than Ctrl so we don't steal Ctrl+Q from apps
            // that bind it internally (vim, etc.).
            Binding {
                mods: MOD_ALT,
                key: key_from_name("Q").unwrap(),
                action: Action::CloseWindow,
            },
        ],
        remaps: vec![Remap {
            from: key_from_name("CapsLock").unwrap(),
            to: key_from_name("Escape").unwrap(),
        }],
        ..Config::default()
    }
}

enum Entry {
    Bind(Binding),
    Remap(Remap),
    ExecOnce(ExecCommand),
    GapsIn(i32),
    GapsOut(i32),
    BorderWidth(i32),
    BorderColor([f32; 4]),
    IdleTimeout(u32),
    Mode(ModePref),
    KeyboardRepeatRate(i32),
    KeyboardRepeatDelay(i32),
    AnimationDurationMs(u32),
    AnimationCurve(AnimationCurve),
    WorkspaceAnimation(WorkspaceAnimation),
    WorkspaceAnimationDurationMs(u32),
    InactiveAlpha(f32),
    InactiveBlur(bool),
    InactiveBlurPasses(u32),
    InactiveBlurRadius(f32),
}

pub fn parse(text: &str) -> Config {
    // Start from the defaults so unset numeric knobs keep sensible
    // values even on a partial config; `bindings` / `remaps` are reset
    // to empty so a user config file fully replaces the built-in
    // bindings rather than appending to them (otherwise the defaults
    // would shadow user overrides with the same mod+key).
    let mut cfg = Config::default();
    cfg.bindings.clear();
    cfg.remaps.clear();
    cfg.exec_once.clear();
    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        match parse_line(line) {
            Ok(Some(Entry::Bind(b))) => cfg.bindings.push(b),
            Ok(Some(Entry::Remap(r))) => cfg.remaps.push(r),
            Ok(Some(Entry::ExecOnce(e))) => cfg.exec_once.push(e),
            Ok(Some(Entry::GapsIn(v))) => cfg.gaps_in = v,
            Ok(Some(Entry::GapsOut(v))) => cfg.gaps_out = v,
            Ok(Some(Entry::BorderWidth(v))) => cfg.border_width = v,
            Ok(Some(Entry::BorderColor(c))) => cfg.border_color = c,
            Ok(Some(Entry::IdleTimeout(v))) => cfg.idle_timeout = v,
            Ok(Some(Entry::Mode(m))) => cfg.mode = Some(m),
            Ok(Some(Entry::KeyboardRepeatRate(v))) => cfg.keyboard_repeat_rate = v,
            Ok(Some(Entry::KeyboardRepeatDelay(v))) => cfg.keyboard_repeat_delay = v,
            Ok(Some(Entry::AnimationDurationMs(v))) => cfg.animation_duration_ms = v,
            Ok(Some(Entry::AnimationCurve(c))) => cfg.animation_curve = c,
            Ok(Some(Entry::WorkspaceAnimation(w))) => cfg.workspace_animation = w,
            Ok(Some(Entry::WorkspaceAnimationDurationMs(v))) => {
                cfg.workspace_animation_duration_ms = v
            }
            Ok(Some(Entry::InactiveAlpha(v))) => cfg.inactive_alpha = v,
            Ok(Some(Entry::InactiveBlur(v))) => cfg.inactive_blur = v,
            Ok(Some(Entry::InactiveBlurPasses(v))) => cfg.inactive_blur_passes = v,
            Ok(Some(Entry::InactiveBlurRadius(v))) => cfg.inactive_blur_radius = v,
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(line = lineno + 1, error = %e, "config: skipping line");
            }
        }
    }
    cfg
}

/// Strip trailing `#` comments but keep `#` that appear inside an exec
/// argument quoted with whitespace context (we don't have quoting yet,
/// so for now this just splits on the first `#`). Good enough for the
/// current feature set.
fn strip_comment(line: &str) -> &str {
    match line.find('#') {
        Some(i) => &line[..i],
        None => line,
    }
}

fn parse_line(line: &str) -> Result<Option<Entry>> {
    let (lhs, rhs) = line
        .split_once('=')
        .ok_or_else(|| anyhow!("expected `key = value`"))?;
    let rhs = rhs.trim();
    match lhs.trim() {
        "bind" => Ok(Some(Entry::Bind(parse_bind(rhs)?))),
        "remap" => Ok(Some(Entry::Remap(parse_remap(rhs)?))),
        "exec-once" => Ok(Some(Entry::ExecOnce(parse_exec(rhs)?))),
        "gaps_in" => Ok(Some(Entry::GapsIn(parse_int(rhs)?))),
        "gaps_out" => Ok(Some(Entry::GapsOut(parse_int(rhs)?))),
        "border_width" => Ok(Some(Entry::BorderWidth(parse_int(rhs)?))),
        "border_color" => Ok(Some(Entry::BorderColor(parse_color(rhs)?))),
        "idle_timeout" => Ok(Some(Entry::IdleTimeout(parse_uint(rhs)?))),
        "mode" => Ok(Some(Entry::Mode(parse_mode(rhs)?))),
        "keyboard_repeat_rate" => Ok(Some(Entry::KeyboardRepeatRate(parse_int(rhs)?))),
        "keyboard_repeat_delay" => Ok(Some(Entry::KeyboardRepeatDelay(parse_int(rhs)?))),
        "animation_duration_ms" => Ok(Some(Entry::AnimationDurationMs(parse_uint(rhs)?))),
        "animation_curve" => Ok(Some(Entry::AnimationCurve(parse_animation_curve(rhs)?))),
        "workspace_animation" => {
            Ok(Some(Entry::WorkspaceAnimation(parse_workspace_animation(rhs)?)))
        }
        "workspace_animation_duration_ms" => {
            Ok(Some(Entry::WorkspaceAnimationDurationMs(parse_uint(rhs)?)))
        }
        "inactive_alpha" => Ok(Some(Entry::InactiveAlpha(parse_float01(rhs)?))),
        "inactive_blur" => Ok(Some(Entry::InactiveBlur(parse_bool(rhs)?))),
        "inactive_blur_passes" => Ok(Some(Entry::InactiveBlurPasses(parse_uint(rhs)?))),
        "inactive_blur_radius" => Ok(Some(Entry::InactiveBlurRadius(parse_pos_float(rhs)?))),
        other => bail!("unknown directive `{other}`"),
    }
}

fn parse_int(s: &str) -> Result<i32> {
    s.parse::<i32>()
        .with_context(|| format!("expected an integer, got `{s}`"))
}

fn parse_uint(s: &str) -> Result<u32> {
    s.parse::<u32>()
        .with_context(|| format!("expected a non-negative integer, got `{s}`"))
}

/// Parse a float in `[0, 1]`. Used for alpha-style knobs where values
/// outside the unit range are nonsense (negative alpha, >1.0 alpha
/// would saturate the same as 1.0). Strict so a typo like `8.5`
/// instead of `0.85` fails loud.
fn parse_float01(s: &str) -> Result<f32> {
    let v = s
        .parse::<f32>()
        .with_context(|| format!("expected a float, got `{s}`"))?;
    if !(0.0..=1.0).contains(&v) {
        bail!("expected a float in [0, 1], got `{s}`");
    }
    Ok(v)
}

/// Parse a non-negative float. Used for blur radius — negative
/// kernel scaling is meaningless and most likely a typo.
fn parse_pos_float(s: &str) -> Result<f32> {
    let v = s
        .parse::<f32>()
        .with_context(|| format!("expected a float, got `{s}`"))?;
    if v < 0.0 || !v.is_finite() {
        bail!("expected a non-negative finite float, got `{s}`");
    }
    Ok(v)
}

fn parse_bool(s: &str) -> Result<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "on" | "yes" | "1" => Ok(true),
        "false" | "off" | "no" | "0" => Ok(false),
        other => bail!("expected a boolean (on/off/true/false), got `{other}`"),
    }
}

fn parse_color(s: &str) -> Result<[f32; 4]> {
    // Accept `RRGGBB`, `#RRGGBB`, or `0xRRGGBB`. Case-insensitive hex.
    let s = s.strip_prefix('#').or_else(|| s.strip_prefix("0x")).unwrap_or(s);
    if s.len() != 6 {
        bail!("expected 6 hex digits for color, got `{s}`");
    }
    let hex = u32::from_str_radix(s, 16)
        .with_context(|| format!("invalid hex color `{s}`"))?;
    Ok(hex_to_rgba(hex))
}

fn parse_exec(rhs: &str) -> Result<ExecCommand> {
    let mut tokens = tokenize_command(rhs)?.into_iter();
    let program = tokens
        .next()
        .context("`exec-once` needs a program name")?;
    let args = tokens.collect();
    Ok(ExecCommand { program, args })
}

/// Shell-ish tokenizer for `exec` / `exec-once` argument lists.
/// Splits on whitespace but treats matching single or double quotes
/// as a group boundary, so things like
///
///   exec-once = sh -c "swww-daemon & exec ~/scripts/wp.sh"
///
/// work as a single three-argument command (sh, -c, <rest>). No
/// escape handling, no variable expansion — just quote grouping,
/// which is what the 90%-case `sh -c "..."` pattern wants. A
/// mismatched quote is an error so typos fail loud instead of
/// swallowing the rest of the line.
fn tokenize_command(s: &str) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut has_content = false;
    for c in s.chars() {
        if in_single {
            if c == '\'' {
                in_single = false;
            } else {
                cur.push(c);
            }
        } else if in_double {
            if c == '"' {
                in_double = false;
            } else {
                cur.push(c);
            }
        } else if c == '\'' {
            in_single = true;
            has_content = true;
        } else if c == '"' {
            in_double = true;
            has_content = true;
        } else if c.is_whitespace() {
            if has_content {
                tokens.push(std::mem::take(&mut cur));
                has_content = false;
            }
        } else {
            cur.push(c);
            has_content = true;
        }
    }
    if in_single || in_double {
        bail!("unterminated quoted string in exec args");
    }
    if has_content {
        tokens.push(cur);
    }
    Ok(tokens)
}

/// Parse a `WIDTHxHEIGHT@HZ` mode spec — same grammar as the SOL_MODE
/// env var. Whitespace around the components is tolerated; `x` and
/// `@` are required separators.
fn parse_mode(rhs: &str) -> Result<ModePref> {
    let (wh, hz) = rhs
        .split_once('@')
        .ok_or_else(|| anyhow!("`mode` expects WIDTHxHEIGHT@HZ"))?;
    let (w, h) = wh
        .split_once('x')
        .ok_or_else(|| anyhow!("`mode` expects WIDTHxHEIGHT@HZ"))?;
    Ok(ModePref {
        width: w.trim().parse().context("invalid mode width")?,
        height: h.trim().parse().context("invalid mode height")?,
        refresh_hz: hz.trim().parse().context("invalid mode refresh")?,
    })
}

fn parse_workspace_animation(s: &str) -> Result<WorkspaceAnimation> {
    match s.trim().to_ascii_lowercase().as_str() {
        "none" | "off" => Ok(WorkspaceAnimation::None),
        "crossfade" | "fade" => Ok(WorkspaceAnimation::Crossfade),
        other => bail!(
            "unknown workspace_animation `{other}` (expected none / crossfade)"
        ),
    }
}

fn parse_animation_curve(s: &str) -> Result<AnimationCurve> {
    // Snake_case names follow easings.net taxonomy. `cubic` and
    // `cubic_out` are aliases for the default since most users mean
    // ease-out when they say "cubic" in a window-manager context.
    match s.trim().to_ascii_lowercase().as_str() {
        "linear" => Ok(AnimationCurve::Linear),
        "cubic" | "cubic_out" => Ok(AnimationCurve::CubicOut),
        "quart" | "quart_out" => Ok(AnimationCurve::QuartOut),
        "quint" | "quint_out" => Ok(AnimationCurve::QuintOut),
        "expo" | "expo_out" => Ok(AnimationCurve::ExpoOut),
        "in_out_cubic" | "cubic_in_out" => Ok(AnimationCurve::InOutCubic),
        other => bail!(
            "unknown animation_curve `{other}` \
             (expected linear / cubic / quart / quint / expo / in_out_cubic)"
        ),
    }
}

fn parse_remap(rhs: &str) -> Result<Remap> {
    let parts: Vec<&str> = rhs.splitn(2, ',').map(|s| s.trim()).collect();
    if parts.len() != 2 {
        bail!("`remap` expects `FROM, TO`");
    }
    let from = key_from_name(parts[0])
        .ok_or_else(|| anyhow!("unknown key `{}`", parts[0]))?;
    let to = key_from_name(parts[1])
        .ok_or_else(|| anyhow!("unknown key `{}`", parts[1]))?;
    Ok(Remap { from, to })
}

fn parse_bind(rhs: &str) -> Result<Binding> {
    // RHS is MODS, KEY, ACTION[, ARGS]. `splitn(4, ',')` leaves commas
    // inside the ARGS portion intact, so `exec, foo --flag=a,b` works.
    // ARGS is optional: actions like `toggle_zoom` take none.
    let parts: Vec<&str> = rhs.splitn(4, ',').map(|s| s.trim()).collect();
    if parts.len() < 3 {
        bail!(
            "`bind` expects `MODS, KEY, ACTION[, ARGS]` (got {} parts)",
            parts.len()
        );
    }
    let mods = parse_mods(parts[0])?;
    let key = key_from_name(parts[1])
        .ok_or_else(|| anyhow!("unknown key `{}`", parts[1]))?;
    let args = parts.get(3).copied().unwrap_or("");
    let action = parse_action(parts[2], args)?;
    Ok(Binding { mods, key, action })
}

fn parse_mods(s: &str) -> Result<u8> {
    let mut bits = 0u8;
    for part in s.split('+') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        bits |= match part.to_ascii_uppercase().as_str() {
            "ALT" | "MOD1" => MOD_ALT,
            "CTRL" | "CONTROL" => MOD_CTRL,
            "SHIFT" => MOD_SHIFT,
            "SUPER" | "META" | "MOD4" => MOD_SUPER,
            other => bail!("unknown modifier `{other}`"),
        };
    }
    Ok(bits)
}

fn parse_action(kind: &str, args: &str) -> Result<Action> {
    match kind {
        // Whitespace-split for now. Quoted arguments would need a
        // real tokenizer; punt on that until a user actually needs
        // filenames with spaces.
        "exec" => Ok(Action::Spawn(parse_exec(args)?)),
        "focus_dir" => Ok(Action::FocusDir(parse_direction(args)?)),
        "move_dir" => Ok(Action::MoveDir(parse_direction(args)?)),
        "toggle_zoom" => Ok(Action::ToggleZoom),
        "close_window" => Ok(Action::CloseWindow),
        "workspace" => Ok(Action::Workspace(parse_workspace_num(args)?)),
        "move_to_workspace" => Ok(Action::MoveToWorkspace(parse_workspace_num(args)?)),
        other => bail!("unknown action `{other}`"),
    }
}

/// Workspace numbers are u32 with a guarded 1-based minimum so
/// `workspace, 0` is a loud error rather than a silent off-by-one.
/// No upper bound: the state vector is sparse (HashMap-ish via
/// `Window.workspace == N` filtering), so binding to workspace 42
/// is legal if pointless.
fn parse_workspace_num(s: &str) -> Result<u32> {
    let n: u32 = s
        .trim()
        .parse()
        .with_context(|| format!("expected a workspace number, got `{s}`"))?;
    if n == 0 {
        bail!("workspace numbers start at 1");
    }
    Ok(n)
}

fn parse_direction(s: &str) -> Result<Direction> {
    // Spelled out, not single-letter: the vim shorthand `l = right`
    // conflicts with "l = left" and would be a foot-gun either way.
    match s.trim().to_ascii_lowercase().as_str() {
        "left" => Ok(Direction::Left),
        "right" => Ok(Direction::Right),
        "up" => Ok(Direction::Up),
        "down" => Ok(Direction::Down),
        other => bail!("unknown direction `{other}` (expected left/right/up/down)"),
    }
}

pub fn mods_to_label(mods: u8) -> String {
    let mut parts: Vec<&str> = Vec::new();
    if mods & MOD_CTRL != 0 {
        parts.push("Ctrl");
    }
    if mods & MOD_ALT != 0 {
        parts.push("Alt");
    }
    if mods & MOD_SHIFT != 0 {
        parts.push("Shift");
    }
    if mods & MOD_SUPER != 0 {
        parts.push("Super");
    }
    parts.join("+")
}

/// Case-insensitive name → evdev scancode lookup. Returns None for
/// anything we don't know yet; the caller turns that into a per-line
/// warning rather than a hard failure.
fn key_from_name(name: &str) -> Option<u32> {
    let up = name.to_ascii_uppercase();
    let code = match up.as_str() {
        "RETURN" | "ENTER" => 28,
        "ESCAPE" | "ESC" => 1,
        "TAB" => 15,
        "SPACE" => 57,
        "BACKSPACE" => 14,
        "CAPSLOCK" | "CAPS_LOCK" | "CAPS" => 58,
        "DELETE" | "DEL" => 111,
        "HOME" => 102,
        "END" => 107,
        "PAGEUP" => 104,
        "PAGEDOWN" => 109,
        "INSERT" | "INS" => 110,
        "LEFT" => 105,
        "RIGHT" => 106,
        "UP" => 103,
        "DOWN" => 108,
        "F1" => 59,
        "F2" => 60,
        "F3" => 61,
        "F4" => 62,
        "F5" => 63,
        "F6" => 64,
        "F7" => 65,
        "F8" => 66,
        "F9" => 67,
        "F10" => 68,
        "F11" => 87,
        "F12" => 88,
        "0" => 11,
        "1" => 2,
        "2" => 3,
        "3" => 4,
        "4" => 5,
        "5" => 6,
        "6" => 7,
        "7" => 8,
        "8" => 9,
        "9" => 10,
        "A" => 30,
        "B" => 48,
        "C" => 46,
        "D" => 32,
        "E" => 18,
        "F" => 33,
        "G" => 34,
        "H" => 35,
        "I" => 23,
        "J" => 36,
        "K" => 37,
        "L" => 38,
        "M" => 50,
        "N" => 49,
        "O" => 24,
        "P" => 25,
        "Q" => 16,
        "R" => 19,
        "S" => 31,
        "T" => 20,
        "U" => 22,
        "V" => 47,
        "W" => 17,
        "X" => 45,
        "Y" => 21,
        "Z" => 44,
        _ => return None,
    };
    Some(code)
}
