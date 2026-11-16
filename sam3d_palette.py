from comfy_api.latest import io


class SWAN_SAM3DTrackPalette(io.ComfyNode):
    """Optional: pin one colour per tracked person, with as many rows as you like.

    Both Swan SAM3DBody Solid Colors (video) and Swan SAM3D Track Colors (GLB)
    take a comma separated hex list and fall back to an automatic even-hue spread
    when it is empty. So this node is entirely optional: leave it unwired and any
    number of people gets a distinct colour automatically. Wire it in only when
    you want to dictate the colours yourself.

    Rows default to white; add or remove them freely. Rows past the tracked count
    are ignored, and missing rows repeat.
    """

    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            io.Color.Input("color", default="#ffffff",
                           tooltip="One colour per tracked person, in track order."),
            prefix="color", min=4, max=64,
        )
        return io.Schema(
            node_id="SWAN_SAM3DTrackPalette",
            display_name="Swan SAM3D Track Palette",
            category="SwanBits/3D",
            description=(
                "Optional colour overrides for tracked people. Wire the hex list into "
                "both Swan SAM3D colour nodes; leave unwired for automatic colours."
            ),
            inputs=[io.Autogrow.Input("colors", template=template)],
            outputs=[io.String.Output(display_name="colors")],
        )

    @classmethod
    def execute(cls, colors) -> io.NodeOutput:
        picked = [value for value in colors.values() if value]
        return io.NodeOutput(",".join(picked))
