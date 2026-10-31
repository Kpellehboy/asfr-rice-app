
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_IMAGE = Path(
    r"C:\Users\Menuo\Desktop\KBK\Papers\Capstone_Research"
    r"\Dataset\archive\Rice_Leaf_AUG\Bacterial Leaf Blight"
    r"\20231006_164208.jpg"
)

OUTPUT_MASK = (
    PROJECT_ROOT / "backend" / "tests" / "assets"
    / "leaf_background_mask.png"
)

DISPLAY_MAX_WIDTH = 850
DISPLAY_MAX_HEIGHT = 700
BRUSH_SIZE = 18  # Brush diameter in original-image pixels


class LeafMaskEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Rice Leaf Mask Annotation")

        if not SOURCE_IMAGE.is_file():
            messagebox.showerror(
                "Image not found",
                f"Check SOURCE_IMAGE path:\n{SOURCE_IMAGE}"
            )
            root.destroy()
            return

        self.original = Image.open(SOURCE_IMAGE).convert("RGB")
        self.width, self.height = self.original.size

        scale = min(
            DISPLAY_MAX_WIDTH / self.width,
            DISPLAY_MAX_HEIGHT / self.height,
            1.0,
        )
        self.display_width = round(self.width * scale)
        self.display_height = round(self.height * scale)
        self.scale = scale

        # Binary mask: 0 = background, 255 = leaf.
        self.mask = Image.new(
            "L", (self.width, self.height), 0
        )
        self.draw = ImageDraw.Draw(self.mask)

        self.brush = BRUSH_SIZE
        self.paint_value = 255
        self.previous_point = None

        self.photo = ImageTk.PhotoImage(
            self.original.resize(
                (self.display_width, self.display_height)
            )
        )

        toolbar = tk.Frame(root)
        toolbar.pack(fill="x", padx=8, pady=6)

        tk.Button(
            toolbar,
            text="Paint LEAF (white)",
            command=lambda: self.set_brush(255),
        ).pack(side="left", padx=3)

        tk.Button(
            toolbar,
            text="Paint BACKGROUND (black)",
            command=lambda: self.set_brush(0),
        ).pack(side="left", padx=3)

        tk.Button(
            toolbar,
            text="Undo last stroke",
            command=self.undo,
        ).pack(side="left", padx=3)

        tk.Button(
            toolbar,
            text="Save mask",
            command=self.save,
        ).pack(side="left", padx=3)

        self.status = tk.Label(
            root,
            text=(
                "Paint visible leaves white. "
                "Leave soil and gaps black. "
                "Drag the mouse to paint."
            ),
            anchor="w",
        )
        self.status.pack(fill="x", padx=8)

        self.canvas = tk.Canvas(
            root,
            width=self.display_width,
            height=self.display_height,
            cursor="crosshair",
        )
        self.canvas.pack(padx=8, pady=8)

        self.canvas.create_image(
            0, 0, image=self.photo, anchor="nw"
        )

        self.stroke_history = []
        self.stroke_points = []

        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        root.protocol("WM_DELETE_WINDOW", self.close)

    def set_brush(self, value):
        self.paint_value = value
        color = "white" if value == 255 else "black"
        self.status.config(
            text=f"Current brush: {color}. Drag over the image."
        )

    def image_point(self, event):
        x = round(event.x / self.scale)
        y = round(event.y / self.scale)

        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        return x, y

    def start_stroke(self, event):
        point = self.image_point(event)
        self.previous_point = point
        self.stroke_points = [point]
        self.apply_line(point, point)

    def paint(self, event):
        point = self.image_point(event)
        if self.previous_point is None:
            return

        self.apply_line(self.previous_point, point)
        self.stroke_points.append(point)
        self.previous_point = point

    def apply_line(self, start, end):
        self.draw.line(
            [start, end],
            fill=self.paint_value,
            width=self.brush,
        )
        radius = self.brush // 2
        x, y = end
        self.draw.ellipse(
            (x-radius, y-radius, x+radius, y+radius),
            fill=self.paint_value,
        )

        # Display the current stroke on the canvas.
        color = "white" if self.paint_value == 255 else "black"
        self.canvas.create_line(
            start[0] * self.scale,
            start[1] * self.scale,
            end[0] * self.scale,
            end[1] * self.scale,
            fill=color,
            width=max(1, self.brush * self.scale),
            capstyle=tk.ROUND,
            smooth=True,
        )

    def end_stroke(self, event):
        if self.stroke_points:
            self.stroke_history.append(
                (self.stroke_points[:], self.paint_value)
            )
        self.previous_point = None
        self.stroke_points = []

    def undo(self):
        if not self.stroke_history:
            return

        # Rebuild the mask from remaining strokes.
        self.stroke_history.pop()
        self.mask = Image.new(
            "L", (self.width, self.height), 0
        )
        self.draw = ImageDraw.Draw(self.mask)

        self.canvas.delete("all")
        self.canvas.create_image(
            0, 0, image=self.photo, anchor="nw"
        )

        for points, value in self.stroke_history:
            self.paint_value = value
            for i, point in enumerate(points):
                start = points[max(0, i - 1)]
                self.apply_line(start, point)

        self.paint_value = 255

    def save(self):
        OUTPUT_MASK.parent.mkdir(parents=True, exist_ok=True)
        self.mask.save(OUTPUT_MASK, format="PNG")

        # Reopen and verify the saved file.
        with Image.open(OUTPUT_MASK) as check:
            check.load()
            values = set(check.convert("L").getdata())

            if check.size != (self.width, self.height):
                messagebox.showerror(
                    "Validation failed",
                    "Mask dimensions do not match the source image.",
                )
                return

        if not values.issubset({0, 255}):
            messagebox.showerror(
                "Validation failed",
                "Mask contains values other than black and white.",
            )
            return

        if 255 not in values:
            messagebox.showwarning(
                "Mask is empty",
                "No leaf pixels were painted white yet.",
            )
            return

        messagebox.showinfo(
            "Mask saved",
            f"Saved and verified:\n{OUTPUT_MASK}\n\n"
            f"Dimensions: {self.width} × {self.height}\n"
            "Black = background; white = leaf.",
        )

    def close(self):
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LeafMaskEditor(root)
    root.mainloop()
