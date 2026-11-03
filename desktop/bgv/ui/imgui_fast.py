"""Faster draw-data replay for wgpu-py's imgui backend.

wgpu 0.32's `ImguiWgpuBackend.render` is meant to cache the texture bind group on the texture view
(`tex_view.__imgui_bind_group`), but inside the class that attribute name is mangled on write and not on read
(`getattr(tex_view, "__imgui_bind_group")`), so the cache never hits and a bind group is created for EVERY draw command of
EVERY frame (about 25 per frame here, ~3 ms of Python + driver time).  `patch(renderer)` replaces the instance's `render`
with the same logic plus a real cache, and skips redundant scissor / bind-group changes.  If anything in the fast path
raises, the original implementation takes over for good.
"""
from __future__ import annotations

import types

import wgpu
from imgui_bundle import imgui


def _fast_render(self, draw_data, render_pass):
    self._clear_pending_textures()
    if draw_data is None:
        return
    display_width, display_height = draw_data.display_size
    fb_width = int(display_width * draw_data.framebuffer_scale.x)
    fb_height = int(display_height * draw_data.framebuffer_scale.y)
    if fb_width <= 0 or fb_height <= 0 or draw_data.cmd_lists_count == 0:
        return
    if draw_data.textures is not None:
        for tex in draw_data.textures:
            if tex.status != imgui.ImTextureStatus.ok:
                self._update_texture(tex)
    self._update_vertex_buffer(draw_data)
    self._set_render_state(draw_data)
    render_pass.set_viewport(0, 0, fb_width, fb_height, 0, 1)
    render_pass.set_pipeline(self._render_pipeline)
    render_pass.set_vertex_buffer(0, self._vertex_buffer)
    index_fmt = wgpu.IndexFormat.uint16 if imgui.INDEX_SIZE == 2 else wgpu.IndexFormat.uint32
    render_pass.set_index_buffer(self._index_buffer, index_fmt, 0)
    render_pass.set_bind_group(0, self._bind_group)
    render_pass.set_blend_constant((0.0, 0.0, 0.0, 0.0))

    cache = self._fast_bg_cache
    scale = draw_data.framebuffer_scale
    off = draw_data.display_pos
    sx, sy, ox, oy = scale.x, scale.y, off.x, off.y
    gv = gi = 0
    last_tex = None
    last_scissor = None
    for commands in draw_data.cmd_lists:
        for command in commands.cmd_buffer:
            tex_id = command.tex_ref.get_tex_id()
            if tex_id != last_tex:
                view = self._texture_views[tex_id]
                hit = cache.get(tex_id)
                if hit is None or hit[0] is not view:
                    bg = self._device.create_bind_group(layout=self._texture_bind_group_layout,
                                                        entries=[{"binding": 0, "resource": view}])
                    cache[tex_id] = hit = (view, bg)
                render_pass.set_bind_group(1, hit[1])
                last_tex = tex_id
            c = command.clip_rect
            x0 = max(int((c.x - ox) * sx), 0)
            y0 = max(int((c.y - oy) * sy), 0)
            x1 = min(int((c.z - ox) * sx), fb_width)
            y1 = min(int((c.w - oy) * sy), fb_height)
            if x1 - x0 <= 0 or y1 - y0 <= 0:
                continue
            sc = (x0, y0, x1 - x0, y1 - y0)
            if sc != last_scissor:
                render_pass.set_scissor_rect(*sc)
                last_scissor = sc
            render_pass.draw_indexed(command.elem_count, 1, command.idx_offset + gi, command.vtx_offset + gv, 0)
        gv += commands.vtx_buffer.size()
        gi += commands.idx_buffer.size()


def _replay(renderer) -> bool:
    """Draw the previous frame's imgui draw data again (no new_frame: the UI code does not run).  False when that is not
    possible (no data yet, window resized)."""
    be = renderer.backend
    imgui.set_current_context(renderer.imgui_context)
    cc = renderer._canvas_context
    dd = imgui.get_draw_data()
    if dd is None or dd.cmd_lists_count == 0:
        return False
    lw, lh = cc.logical_size
    if abs(dd.display_size.x - lw) > 0.5 or abs(dd.display_size.y - lh) > 0.5:
        return False
    enc = be._device.create_command_encoder()
    view = cc.get_current_texture().create_view()
    rp = enc.begin_render_pass(color_attachments=[{"view": view, "resolve_target": None, "clear_value": (0, 0, 0, 1),
                                                   "load_op": wgpu.LoadOp.load, "store_op": wgpu.StoreOp.store}])
    be.render(dd, rp)
    rp.end()
    be._device.queue.submit([enc.finish()])
    return True


def patch(renderer, lite=None) -> bool:
    """Install the fast replay on `renderer.backend` (idempotent).  Returns True when installed.

    `lite`: optional callable returning True when THIS frame should reuse the previous frame's draw data instead of
    rebuilding the UI (the UI manager asks for it only when the app is below its frame-rate target)."""
    try:
        be = renderer.backend
        if getattr(be, "_fast_installed", False):
            holder = getattr(renderer, "_fast_lite", None)
            if holder is not None and lite is not None:
                holder[0] = lite                      # a new UI manager (domain switch): its lite_frame takes over
            return True
        orig = be.render
        be._fast_bg_cache = {}
        state = {"ok": True}

        def render(draw_data, render_pass):
            if state["ok"]:
                try:
                    return _fast_render(be, draw_data, render_pass)
                except Exception:
                    state["ok"] = False
            return orig(draw_data, render_pass)

        be.render = render
        be._fast_installed = True
        if lite is not None:
            orig_frame = renderer.render
            holder = renderer._fast_lite = [lite]

            def render_frame():
                try:
                    if holder[0]() and _replay(renderer):
                        return
                except Exception:
                    pass
                return orig_frame()

            renderer.render = render_frame
        return True
    except Exception:
        return False
