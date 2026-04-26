//! xkbcommon keymap + state, plus the memfd we pass to clients so they
//! don't need to re-negotiate the layout on their side.
//!
//! Hardcoded to US layout with no variant at B5. Real config comes in B8.

use std::io::Write;
use std::os::fd::OwnedFd;

use anyhow::{Context, Result, anyhow};
use rustix::fs::{MemfdFlags, ftruncate, memfd_create};
use xkbcommon::xkb;

/// Wayland expects evdev keycodes offset by 8 when indexing an xkb keymap
/// (a historical X11 convention). libinput gives us raw evdev codes.
pub const WL_KEY_OFFSET: u32 = 8;

pub struct KeymapState {
    pub keymap: xkb::Keymap,
    pub state: xkb::State,
    pub fd: OwnedFd,
    pub size: u32,
    last_mods: ModifiersSnapshot,
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct ModifiersSnapshot {
    pub depressed: u32,
    pub latched: u32,
    pub locked: u32,
    pub group: u32,
}

impl KeymapState {
    pub fn new_us() -> Result<Self> {
        let context = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = xkb::Keymap::new_from_names(
            &context,
            "",
            "",
            "us",
            "",
            None,
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .ok_or_else(|| anyhow!("xkb_keymap_new_from_names returned null"))?;
        let state = xkb::State::new(&keymap);

        let text = keymap.get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1);
        let bytes = text.as_bytes();
        let size = bytes.len() as u32;

        let fd = memfd_create("sol-keymap", MemfdFlags::CLOEXEC).context("memfd_create keymap")?;
        ftruncate(&fd, size as u64).context("ftruncate keymap")?;
        {
            // Use the same fd via a File to write; we keep the OwnedFd around.
            let mut file = std::fs::File::from(fd.try_clone().context("dup keymap fd")?);
            file.write_all(bytes).context("write keymap")?;
        }

        Ok(Self {
            keymap,
            state,
            fd,
            size,
            last_mods: ModifiersSnapshot::default(),
        })
    }

    /// Feed a key press/release to the xkb state machine and return the
    /// current modifier snapshot (for wl_keyboard.modifiers).
    pub fn feed_key(&mut self, evdev_keycode: u32, pressed: bool) -> ModifiersSnapshot {
        let direction = if pressed {
            xkb::KeyDirection::Down
        } else {
            xkb::KeyDirection::Up
        };
        self.state
            .update_key((evdev_keycode + WL_KEY_OFFSET).into(), direction);
        ModifiersSnapshot {
            depressed: self.state.serialize_mods(xkb::STATE_MODS_DEPRESSED),
            latched: self.state.serialize_mods(xkb::STATE_MODS_LATCHED),
            locked: self.state.serialize_mods(xkb::STATE_MODS_LOCKED),
            group: self.state.serialize_layout(xkb::STATE_LAYOUT_EFFECTIVE),
        }
    }

    /// True if the modifier state has changed since the last call;
    /// caller uses this to decide whether to emit wl_keyboard.modifiers.
    pub fn mods_changed(&mut self, current: ModifiersSnapshot) -> bool {
        if self.last_mods == current {
            false
        } else {
            self.last_mods = current;
            true
        }
    }
}
