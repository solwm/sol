//! `ext_workspace_v1` — portable workspace protocol for panels.
//!
//! Waybar's `wlr/workspaces` module speaks this: it binds the
//! manager global, discovers our workspace list, shows them in the
//! bar, and sends `activate` when the user clicks one. We reply
//! with `state(active)` + `done` on switch so the bar highlights
//! the current workspace.
//!
//! Protocol shape (for our single-output, fixed-count setup):
//!   ext_workspace_manager_v1 — the global
//!     └─ one ext_workspace_group_handle_v1 (covers our only output)
//!          └─ five ext_workspace_handle_v1 (named "1".."5", ids "ws-1".."ws-5")
//!
//! Per-client bindings are independent — each binder gets its own
//! group + workspace handles. On `switch_workspace` we walk every
//! live binding and emit state changes, then `done` to commit the
//! change atomically.
//!
//! We advertise only the `activate` workspace capability; no
//! create/remove/assign (workspaces are fixed). `urgent` and
//! `hidden` bits aren't used yet — they'd want a protocol-level
//! concept of "this client wants attention" that we don't track.

use wayland_protocols::ext::workspace::v1::server::{
    ext_workspace_group_handle_v1::{
        self, ExtWorkspaceGroupHandleV1, GroupCapabilities,
    },
    ext_workspace_handle_v1::{self, ExtWorkspaceHandleV1, State as WsState, WorkspaceCapabilities},
    ext_workspace_manager_v1::{self, ExtWorkspaceManagerV1},
};
use wayland_server::{
    Client, DataInit, Dispatch, DisplayHandle, GlobalDispatch, New, Resource,
    protocol::wl_output::WlOutput,
};

/// Fixed workspace count. Matches the five bindings the config
/// advertises (Alt+Y..P); bumping this requires both a config
/// update and a rebuild, but workspace numbering stays 1-based.
pub const NUM_WORKSPACES: u32 = 5;

pub const EXT_WORKSPACE_MANAGER_VERSION: u32 = 1;

/// Per-manager bookkeeping held on `State`. Tracks the server-side
/// resources we created when this client bound the manager, so
/// `notify_active_changed` can target them on workspace switches.
/// Resources become inert on client disconnect; we prune on each
/// notify.
pub struct ManagerBinding {
    pub manager: ExtWorkspaceManagerV1,
    pub group: ExtWorkspaceGroupHandleV1,
    /// One per workspace, in 1-based order: `workspaces[0]` is ws 1.
    pub workspaces: Vec<ExtWorkspaceHandleV1>,
}

/// UserData on each `ext_workspace_handle_v1` — carries the
/// 1-based workspace number so the `activate` request can route
/// back to the compositor's `switch_workspace`.
pub struct WorkspaceData {
    pub number: u32,
}

impl GlobalDispatch<ExtWorkspaceManagerV1, ()> for crate::State {
    fn bind(
        state: &mut Self,
        dh: &DisplayHandle,
        client: &Client,
        resource: New<ExtWorkspaceManagerV1>,
        _gd: &(),
        init: &mut DataInit<'_, Self>,
    ) {
        let manager = init.init(resource, ());
        tracing::info!(id = ?manager.id(), "bind ext_workspace_manager_v1");

        // Create the group and emit it *before* any workspaces — the
        // client needs the group resource to receive workspace_enter
        // events in the correct order.
        let group = match client.create_resource::<ExtWorkspaceGroupHandleV1, (), crate::State>(
            dh,
            manager.version(),
            (),
        ) {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!(error = ?e, "ext_workspace: create group failed");
                return;
            }
        };
        manager.workspace_group(&group);
        // No `create_workspace` — our set is fixed at startup.
        group.capabilities(GroupCapabilities::empty());

        // Bind the group to this client's wl_output resources. If
        // the client hasn't bound wl_output yet (unusual — most
        // taskbars bind outputs first), `output_enter` is emitted
        // later when they do, see `notify_output_bound`.
        for output in client_outputs(client, &state.outputs) {
            group.output_enter(&output);
        }

        // Create the N workspace handles and emit them + initial
        // name/coords/state/caps for each.
        let mut workspaces = Vec::with_capacity(NUM_WORKSPACES as usize);
        for n in 1..=NUM_WORKSPACES {
            let ws = match client
                .create_resource::<ExtWorkspaceHandleV1, WorkspaceData, crate::State>(
                    dh,
                    manager.version(),
                    WorkspaceData { number: n },
                )
            {
                Ok(w) => w,
                Err(e) => {
                    tracing::warn!(n, error = ?e, "ext_workspace: create workspace failed");
                    continue;
                }
            };
            manager.workspace(&ws);
            ws.id(format!("ws-{n}"));
            ws.name(format!("{n}"));
            // 1-D coordinates, one dimension, ordering 0..=N-1 so
            // bars can render them in their declared order.
            ws.coordinates((n - 1).to_ne_bytes().to_vec());
            ws.state(state_bits_for(state, n));
            ws.capabilities(WorkspaceCapabilities::Activate);
            group.workspace_enter(&ws);
            workspaces.push(ws);
        }

        // Tell the client "initial burst done" so it renders the
        // bar state in one frame rather than per-event.
        manager.done();

        state.ext_workspace_managers.push(ManagerBinding {
            manager,
            group,
            workspaces,
        });
    }
}

impl Dispatch<ExtWorkspaceManagerV1, ()> for crate::State {
    fn request(
        state: &mut Self,
        _client: &Client,
        resource: &ExtWorkspaceManagerV1,
        request: ext_workspace_manager_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        match request {
            // We apply per-request (activate, etc.) immediately rather
            // than buffering, so commit is effectively a no-op. Clients
            // still have to send it to signal "end of request group"
            // per the protocol contract.
            ext_workspace_manager_v1::Request::Commit => {}
            ext_workspace_manager_v1::Request::Stop => {
                resource.finished();
                state
                    .ext_workspace_managers
                    .retain(|m| m.manager.id() != resource.id());
            }
            _ => {}
        }
    }
}

impl Dispatch<ExtWorkspaceGroupHandleV1, ()> for crate::State {
    fn request(
        _state: &mut Self,
        _client: &Client,
        _resource: &ExtWorkspaceGroupHandleV1,
        request: ext_workspace_group_handle_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        // `create_workspace` is ignored by design: we don't advertise
        // the capability, and the workspace count is fixed. Destroy
        // is a pure protocol bookkeeping op — wayland-server handles
        // cleanup via the destructor annotation in the XML.
        match request {
            ext_workspace_group_handle_v1::Request::CreateWorkspace { .. } => {}
            ext_workspace_group_handle_v1::Request::Destroy => {}
            _ => {}
        }
    }
}

impl Dispatch<ExtWorkspaceHandleV1, WorkspaceData> for crate::State {
    fn request(
        state: &mut Self,
        _client: &Client,
        _resource: &ExtWorkspaceHandleV1,
        request: ext_workspace_handle_v1::Request,
        data: &WorkspaceData,
        _dh: &DisplayHandle,
        _init: &mut DataInit<'_, Self>,
    ) {
        // Route through the normal switch path so keyboard focus,
        // pointer focus, and zoom get the same treatment as a
        // keybind-driven switch. notify fires from inside
        // switch_workspace. Deactivate/assign/remove caps aren't
        // advertised, so other requests are per-spec ignored.
        if let ext_workspace_handle_v1::Request::Activate = request {
            crate::switch_workspace(state, data.number);
        }
    }
}

/// Called from `switch_workspace` after `active_ws` has been updated.
/// Emits `state(0)` on the previously-active handle and `state(active)`
/// on the new one for every live manager binding, then `done` to
/// batch the change atomically.
pub fn notify_active_changed(state: &mut crate::State, old_ws: u32, new_ws: u32) {
    state.ext_workspace_managers.retain(|m| m.manager.is_alive());
    for binding in &state.ext_workspace_managers {
        if let Some(old) = workspace_handle(binding, old_ws) {
            old.state(WsState::empty());
        }
        if let Some(new) = workspace_handle(binding, new_ws) {
            new.state(WsState::Active);
        }
        binding.manager.done();
    }
}

/// Called from `output.rs` whenever a client binds `wl_output`.
/// If this client already bound the workspace manager, emit
/// `output_enter` on their group so the bar knows which monitor
/// these workspaces belong to. Taskbars that bind the manager
/// before the output rely on this path.
pub fn notify_output_bound(state: &crate::State, client: &Client, output: &WlOutput) {
    for binding in &state.ext_workspace_managers {
        let Some(manager_client) = binding.manager.client() else {
            continue;
        };
        if manager_client.id() == client.id() {
            binding.group.output_enter(output);
        }
    }
}

fn workspace_handle(binding: &ManagerBinding, n: u32) -> Option<&ExtWorkspaceHandleV1> {
    if n == 0 || n > NUM_WORKSPACES {
        return None;
    }
    binding.workspaces.get((n - 1) as usize)
}

fn state_bits_for(state: &crate::State, n: u32) -> WsState {
    if state.active_ws == n {
        WsState::Active
    } else {
        WsState::empty()
    }
}

/// Filter `state.outputs` down to the ones owned by the given client.
/// A client may have bound the output more than once, but in practice
/// everyone binds it once, so cardinality is 0 or 1 per client here.
fn client_outputs(client: &Client, outputs: &[WlOutput]) -> Vec<WlOutput> {
    outputs
        .iter()
        .filter(|o| o.client().map(|c| c.id()) == Some(client.id()))
        .cloned()
        .collect()
}
