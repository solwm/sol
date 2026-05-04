//! `wl_subcompositor` + `wl_subsurface` — wire dispatch lives in
//! smithay (`delegate_compositor!`); the role-attachment + parent
//! linkage lives in `CompositorHandler::new_subsurface` and
//! `CompositorHandler::destroyed` in lib.rs.
//!
//! Subsurface position handling (`wl_subsurface.set_position`) is
//! smithay's `SubsurfaceCachedState::location`. We mirror it into
//! our `SolSurfaceData::subsurface_offset` on commit so the scene
//! walker doesn't have to re-enter `with_states` per-frame.
//!
//! Out of scope (kept inherited from the pre-migration shape):
//! `place_above` / `place_below` sibling ordering, subsurface damage
//! tracking. Smithay treats commits as desync by default, which
//! matches our previous "every commit immediate" behaviour.
