import sys
import numpy as np

_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

Here are some examples:

Call it with a chunk-size of 10
$ {sys.argv[0]} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ {sys.argv[0]} 10 100x500

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
spanning the coordinates x \\in 0.1-0.3 and y \\in 0.2-0.3
$ {sys.argv[0]} 10 100x500 0.1:0.3 0.2:0.3
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# First we define all the defaults, then we let the arguments overwrite
# them.
chunk_size = 10
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Now grab the arguments
argv = sys.argv[1:]
if argv:
    chunk_size = int(argv.pop(0))
if argv:
    size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

print(f"""\
Calculating the Mandelbrot set with these arguments:

{chunk_size = }
{size = }
{xlim = }
{ylim = }
""")

# Convert to numpy arrays, not really needed...
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Dimensions of the image
image = np.zeros(size)

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

for x in range(size[0]):
    cx = complex(xlim[0] + x * xconst, 0)
    for y in range(size[1]):
        # process (x, y)
        c = cx + complex(0, ylim[0] + y * yconst)
        z = 0
        for i in range(100):
            z = z*z + c
            if np.abs(z) > 2:
                image[x, y] = i
                break

# Check if we should save plot or just data
if "--output" in sys.argv or "--show" in sys.argv:
    import matplotlib.pyplot as plt

    # Increase font-size
    plt.rcParams.update({
        "font.size": 10,
    })
    plt.imshow(image.T, extent=np.concatenate([xlim, ylim]))
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")

    # Just minimize white-space around the actual plot...
    plt.margins(0, 0)
    plt.savefig("Figure_1.png", bbox_inches="tight", pad_inches=0)
    plt.show()
else:
    # Save the result as numpy array for testing
    output_file = f"baseline_{size[0]}x{size[1]}.npy"
    np.save(output_file, image)
    print(f"Result saved to {output_file}")