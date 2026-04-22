//! User config loader for voidptr.
//!
//! Looks for, in order:
//!   1. `$XDG_CONFIG_HOME/voidptr/voidptr.conf`
//!   2. `$HOME/.config/voidptr/voidptr.conf`
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
//! bind  = MOD[+MOD2], KEY, exec, program with args
//! remap = FROM, TO                  # rewrite a scancode before anything else
//! ```
//!
//! Modifiers (case-insensitive): `ALT`/`MOD1`, `CTRL`/`CONTROL`,
//! `SHIFT`, `SUPER`/`META`/`MOD4`. Keys: letters `A`-`Z`, digits
//! `0`-`9`, `F1`-`F12`, `Return`/`Enter`, `Escape`, `Tab`, `Space`,
//! `Backspace`, `CapsLock`, `Delete`, `Home`, `End`, `PageUp`,
//! `PageDown`, `Insert`, `Left`, `Right`, `Up`, `Down`. Actions:
//! `exec` only for now — future directives (quit, focus, workspace
//! switching, …) plug in at `parse_action`.
//!
//! `remap` rewrites at the evdev scancode layer, so it's invisible to
//! both bindings and clients: the remapped code is what xkb sees and
//! what gets fed to `wl_keyboard.key`. Built-in default remaps
//! CapsLock to Escape because that's what the primary user expects.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};

pub const MOD_ALT: u8 = 1 << 0;
pub const MOD_CTRL: u8 = 1 << 1;
pub const MOD_SHIFT: u8 = 1 << 2;
pub const MOD_SUPER: u8 = 1 << 3;

#[derive(Debug, Clone, Default)]
pub struct Config {
    pub bindings: Vec<Binding>,
    pub remaps: Vec<Remap>,
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
    Spawn { program: String, args: Vec<String> },
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

fn config_path() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("voidptr").join("voidptr.conf");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config/voidptr/voidptr.conf");
    }
    PathBuf::from("voidptr.conf")
}

pub fn default_config() -> Config {
    Config {
        bindings: vec![
            Binding {
                mods: MOD_ALT,
                key: key_from_name("Return").unwrap(),
                action: Action::Spawn {
                    program: "alacritty".to_string(),
                    // `-e zsh` so we get the user's real shell with
                    // their ~/.zshrc applied — otherwise alacritty
                    // defaults to $SHELL which on some setups skips
                    // interactive rc files.
                    args: vec!["-e".to_string(), "zsh".to_string()],
                },
            },
            Binding {
                mods: MOD_ALT,
                key: key_from_name("D").unwrap(),
                action: Action::Spawn {
                    program: "rofi".to_string(),
                    args: vec!["-show".to_string(), "drun".to_string()],
                },
            },
        ],
        remaps: vec![Remap {
            from: key_from_name("CapsLock").unwrap(),
            to: key_from_name("Escape").unwrap(),
        }],
    }
}

enum Entry {
    Bind(Binding),
    Remap(Remap),
}

pub fn parse(text: &str) -> Config {
    let mut bindings = Vec::new();
    let mut remaps = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        match parse_line(line) {
            Ok(Some(Entry::Bind(b))) => bindings.push(b),
            Ok(Some(Entry::Remap(r))) => remaps.push(r),
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(line = lineno + 1, error = %e, "config: skipping line");
            }
        }
    }
    Config { bindings, remaps }
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
    match lhs.trim() {
        "bind" => Ok(Some(Entry::Bind(parse_bind(rhs.trim())?))),
        "remap" => Ok(Some(Entry::Remap(parse_remap(rhs.trim())?))),
        other => bail!("unknown directive `{other}`"),
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
    // RHS is MODS, KEY, ACTION, ARGS. `splitn(4, ',')` leaves commas
    // inside the ARGS portion intact, so `exec, foo --flag=a,b` works.
    let parts: Vec<&str> = rhs.splitn(4, ',').map(|s| s.trim()).collect();
    if parts.len() < 4 {
        bail!(
            "`bind` expects `MODS, KEY, ACTION, ARGS` (got {} parts)",
            parts.len()
        );
    }
    let mods = parse_mods(parts[0])?;
    let key = key_from_name(parts[1])
        .ok_or_else(|| anyhow!("unknown key `{}`", parts[1]))?;
    let action = parse_action(parts[2], parts[3])?;
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
        "exec" => {
            // Whitespace-split for now. Quoted arguments would need a
            // real tokenizer; punt on that until a user actually needs
            // filenames with spaces.
            let mut words = args.split_whitespace();
            let program = words
                .next()
                .context("`exec` needs a program name")?
                .to_string();
            let args = words.map(|s| s.to_string()).collect();
            Ok(Action::Spawn { program, args })
        }
        other => bail!("unknown action `{other}`"),
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
