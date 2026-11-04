import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares





# ============================================================
# Ellipse functions
# ============================================================

def normalise_phi(phi):
    """
    Keep phi in the range -pi/2 to +pi/2.
    """
    return (phi + np.pi / 2) % np.pi - np.pi / 2


def ellipse_points(theta, h, k, x_radius, y_radius, phi):
    """
    Generate points on a rotated ellipse.

    Convention:
        h, k      = ellipse centre
        x_radius  = radius along the local x-like ellipse axis
        y_radius  = radius along the local y-like ellipse axis
        phi       = rotation angle of local x-axis from lab x-axis

    If phi = 0:
        x = h + x_radius*cos(theta)
        y = k + y_radius*sin(theta)
    """

    theta = np.asarray(theta)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    x = h + x_radius * cos_t * cos_p - y_radius * sin_t * sin_p
    y = k + x_radius * cos_t * sin_p + y_radius * sin_t * cos_p

    return x, y


def point_to_ellipse_angle(x, y, h, k, x_radius, y_radius, phi):
    """
    Convert a lab-coordinate point into the ellipse parameter angle theta.
    """

    dx = x - h
    dy = y - k

    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    # Rotate point back into the ellipse's local coordinate frame
    u =  cos_p * dx + sin_p * dy
    v = -sin_p * dx + cos_p * dy

    theta = np.arctan2(v / y_radius, u / x_radius)

    return theta


def fit_ellipse(x, y, phi_bound_deg=45):
    """
    Fit an ellipse using the beamline-style axis convention directly.

    Returns:
        h, k, x_radius, y_radius, phi

    where phi should be close to 0 for a mostly vertical ellipse.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if len(x) < 5:
        raise ValueError("At least 5 points are needed to fit an ellipse.")

    # Initial centre estimate
    h0 = np.mean(x)
    k0 = np.mean(y)

    points = np.column_stack([x, y])
    centred = points - np.array([h0, k0])

    # Estimate main vertical-ish direction from covariance
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Long-axis eigenvector
    long_axis = eigvecs[:, np.argmax(eigvals)]
    long_axis_angle = np.arctan2(long_axis[1], long_axis[0])

    # In this convention, y_radius axis is at phi + 90 degrees.
    # Therefore phi = long_axis_angle - 90 degrees.
    phi0 = long_axis_angle - np.pi / 2
    phi0 = normalise_phi(phi0)

    # Keep phi near beamline-style convention
    phi_bound = np.radians(phi_bound_deg)
    phi0 = np.clip(phi0, -phi_bound, phi_bound)

    cos_p = np.cos(phi0)
    sin_p = np.sin(phi0)

    dx = x - h0
    dy = y - k0

    # Coordinates in local ellipse frame
    u =  cos_p * dx + sin_p * dy
    v = -sin_p * dx + cos_p * dy




    x_radius0 = max((np.max(u) - np.min(u)) / 2, 1e-6)
    y_radius0 = max((np.max(v) - np.min(v)) / 2, 1e-6)

    data_span = max(np.ptp(dx), np.ptp(dy))
    if data_span <= 0:
        data_span = max(abs(x_radius0), abs(y_radius0), 1.0)
    max_radius = data_span * 10.0  # allow some slack
    log_max_radius = np.log(max_radius)

    # Parameters:
    # h, k, log(x_radius), log(y_radius), phi
    p0 = np.array([
        h0,
        k0,
        np.log(x_radius0),
        np.log(y_radius0),
        phi0
    ])

    lower_bounds = [
        -np.inf,
        -np.inf,
        np.log(1e-9),
        np.log(1e-9),
        -phi_bound
    ]

    upper_bounds = [
        np.inf,
        np.inf,
        log_max_radius,
        log_max_radius,
        phi_bound
    ]

    def residuals(p):
        h, k, log_x_radius, log_y_radius, phi = p

        x_radius = np.exp(log_x_radius)
        y_radius = np.exp(log_y_radius)

        dx = x - h
        dy = y - k

        cos_p = np.cos(phi)
        sin_p = np.sin(phi)

        u =  cos_p * dx + sin_p * dy
        v = sin_p * dx - cos_p * dy

        # Points on ellipse satisfy:
        # (u/x_radius)^2 + (v/y_radius)^2 = 1
        return (u / x_radius)**2 + (v / y_radius)**2 - 1

    result = least_squares(
        residuals,
        p0,
        bounds=(lower_bounds, upper_bounds)
    )

    h, k, log_x_radius, log_y_radius, phi = result.x

    x_radius = np.exp(log_x_radius)
    y_radius = np.exp(log_y_radius)
    phi = normalise_phi(phi)

    return h, k, x_radius, y_radius, phi


def fit_and_generate_ellipse(x, y, n_points=360, start_index=0, half_fit = False):
    """
    Fit ellipse and generate n_points around it.

    start_index=0 means start from A, assuming A is the first point in x/y arrays.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    h, k, x_radius, y_radius, phi = fit_ellipse(x, y)

    x_start = x[start_index]
    y_start = y[start_index]

    theta_start = point_to_ellipse_angle(
        x_start,
        y_start,
        h,
        k,
        x_radius,
        y_radius,
        phi
    )
    arc_span = np.pi if half_fit else 2 * np.pi
    theta = theta_start + np.linspace(0, arc_span, n_points, endpoint=False)

    x_ellipse, y_ellipse = ellipse_points(
        theta,
        h,
        k,
        x_radius,
        y_radius,
        phi
    )

    params = h, k, x_radius, y_radius, phi

    # Beamline-style coefficient order
    rotEllCoeff = [
        x_radius,
        y_radius,
        h,
        k,
        phi
    ]

    return x_ellipse, y_ellipse, params, rotEllCoeff



# ============================================================
# Plotting and printing
# ============================================================


def plot_ellipse(x_data, y_data, x_ellipse, y_ellipse, XC, YC, upper=True, n=400):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(x_data, y_data, label="Input points",color = "g", s=60)
    ax.plot(x_ellipse, y_ellipse, label="Fitted ellipse", linewidth=2)
    ax.scatter(XC, YC, marker="x", s=100, label="Centre")
    ax.scatter(x_ellipse[0], y_ellipse[0], marker="*", s=160, label="Start point")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Ellipse fit")
    ax.legend()
    ax.grid(True)
    ax.axis("auto")

def plot_fitted_ellipse(x_data, y_data, n_points=360, start_index=0,half = False):
    """
    Basic plot to check ellipse fit clearly.
    """

    x_ellipse, y_ellipse, params, rotEllCoeff = fit_and_generate_ellipse(
        x_data,
        y_data,
        n_points=n_points,
        start_index=start_index,
        half_fit = half
    )

    h, k, x_radius, y_radius, phi = params

    plot_ellipse(x_data, y_data, x_ellipse, y_ellipse, h, k, upper=True, n=n_points)

    # This makes the plot easy to read rather than geometrically exact
    print("rotEllCoeff = [x_radius, y_radius, x_centre, y_centre, phi]")
    print(f"rotEllCoeff = [{rotEllCoeff[0]:.4f}, {rotEllCoeff[1]:.4f}, {rotEllCoeff[2]:.4f}, {rotEllCoeff[3]:.4f}, {rotEllCoeff[4]:.4f}]")
    # print(rotEllCoeff)
    plt.tight_layout()
    # plt.show()

    

    return x_ellipse, y_ellipse, params, rotEllCoeff

def print_ellipse_lists(x_ellipse, y_ellipse, precision=4):
    """
    Print x_ellipse and y_ellipse as copy-pasteable lists.
    """

    x_list = [round(float(x), precision) for x in x_ellipse]
    y_list = [round(float(y), precision) for y in y_ellipse]

    print("x_ellipse = [")
    for x in x_list:
        print(f"    {x},")
    print("]")

    print("\ny_ellipse = [")
    for y in y_list:
        print(f"    {y},")
    print("]")


# ============================================================
# Placeholder scan function
# ============================================================

def run_ellipse_scan(x, y, SCAN_PARAMS):
    iterations = len(x)

    for i in range(iterations):
        x_pos = x[i]
        y_pos = y[i]

        print(f"Point {i + 1}/{iterations}: x = {x_pos:.4f}, y = {y_pos:.4f}")

        # MOVE MOTOR X
        # move_x_motor(x_pos)

        # MOVE MOTOR Y
        # move_y_motor(y_pos)

        # RUN SCAN COMMAND
        # run_scan(SCAN_PARAMS)

    return

# ============================================================
# Input data
# ============================================================
X_data = [
    -2.42,
    -0.87,
     1.60, #C
     2.60, 
     1.03,
     -1.50,
     2.42,
     0.75,
     -2.32,
     -1.83,
     -0.16,
     2.05
]

Y_data = [
    10.39, #A
    47.19, #b
    33.19, #C
    -14.16, #D
    -46.24, #E
    -32.70, #F
    0.14,
    41.84,
    -12.56,
    37.07,
    -47.61,
    -35.70
]

X_data_2 = [
   2.47,
   1.68,
   -0.87,
   -2.51,
   -1.70,
   0.75,
]

Y_data_2 = [
    1.87,
    -42.24,
    -44.43,
    -3.87,
    39.05,
    42.11
]

# ============================================================
# Run test
# ============================================================
print(len(X_data),len(Y_data))


x_ellipse_2, y_ellipse_2, params, rotEllCoeff = plot_fitted_ellipse(
    X_data_2,
    Y_data_2,
    n_points=360,
    start_index=0,
    half=False
)


def calculate_point(angle, params, x_off=0, y_off=0):

    Xr, Yr, Xc, Yc, phi = params
    # print(angle)
    rad = float(angle) * np.pi / 180
    
    x = x_off + Xc + Xr * np.cos(rad)*np.cos(phi) - Yr * np.sin(rad)*np.sin(phi)
    y = y_off + Yc + Xr * np.cos(rad)*np.sin(phi) + Yr * np.sin(rad)*np.cos(phi)

    x_wrong = x_off + Xc + Xr * np.cos(rad)*np.cos(phi) - Yr * np.sin(rad)*np.sin(phi)
    y_wrong = y_off + Yc + Xr * np.cos(rad)*np.sin(phi) - Yr * np.sin(rad)*np.cos(phi)

    return x, y, x_wrong, y_wrong

angles = np.arange(-5, 365, 1)

# print(angles)

#Plot the fit vs the calculated points from the code

for angle in angles:
    x,y,x_w,y_w = calculate_point(angle, rotEllCoeff)

    plt.scatter(x, y, color='b', s=10, label='Correct - +' if angle == -5 else "")
    # plt.scatter(x_w, y_w, color='r', s=10)
plt.legend()
plt.show()
    









#print_ellipse_lists(x_ellipse, y_ellipse)
# Uncomment this if you want the full generated lists printed
# print_ellipse_lists(x_ellipse, y_ellipse, precision=4)

# Uncomment this when ready to run the actual scan
# SCAN_PARAMS = {}
# run_ellipse_scan(x_ellipse, y_ellipse, SCAN_PARAMS)