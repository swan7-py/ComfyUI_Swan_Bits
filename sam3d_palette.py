from comfy_api.latest import io


class SWAN_SAM3DTrackPalette(io.ComfyNode):
    """Optional: pin one colour per tracked person.

    Both Swan SAM3DBody Solid Colors (video) and Swan SAM3D Track Colors (GLB)
    take a comma separated hex list and fall back to an automatic even-hue spread
    when it is empty, so this node is entirely optional. Leave it unwired and any
    number of people gets a distinct colour for free; wire it in only when you
    want to dictate the colours yourself.

    Rows default to white. Entries past the tracked count are ignored, and a
    shorter list repeats.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SWAN_SAM3DTrackPalette",
            display_name="Swan SAM3D Track Palette",
            category="SwanBits/3D",
            description=(
                "Optional colour overrides for tracked people. Wire the hex list into "
                "both Swan SAM3D colour nodes; leave unwired for automatic colours."
            ),
            inputs=[
                io.Colors.Input(
                    "colors",
                    default=["#ffffff", "#ffffff", "#ffffff", "#ffffff"],
                    tooltip="One colour per tracked person, in track order.",
                ),
            ],
            outputs=[io.String.Output(display_name="colors")],
        )

    @classmethod
    def execute(cls, colors) -> io.NodeOutput:
        # COLORS arrives either as an ordered dict or a plain list depending on
        # how the graph was serialised; accept both rather than guess.
        values = colors.values() if isinstance(colors, dict) else (colors or [])
        picked = [value for value in values if value]
        return io.NodeOutput(",".join(picked))
