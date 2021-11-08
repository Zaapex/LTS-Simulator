import numpy as np
from Vehicle import *
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy import optimize

g = 9.81


def max_velocity_rear(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, W):
    """Calculation of max velocity for rear axle"""

    m_rear = m * b / l

    c2 = (-CoPy * W * alfa_cl * r + CoPy * alfa_cl * r + CoPz * W * alfa_cd * r + CoPz * alfa_cd * r -
          W * alfa_cd * h * r + W * alfa_cl * l * r + 2 * W * h * l * m_rear + W * l * m_rear * w + alfa_cd * h * r
          + alfa_cd * r * w - alfa_cl * l * r + 2 * h * l * m_rear + l * m_rear * w) / (
                     2 * l * r)

    c1 = 0

    c0 = (W * a * g * m - a * g * m) / (2 * l)

    coeff = [c2, c1, c0]
    Y = np.roots(coeff)

    return Y[1]


def normal_force_rear_outer(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, v, d,acc):
    """Calculation of normal force on rear outer tire"""

    m_rear = m * a / l
    massforce = m * g * b
    accforce = acc * m * (h + w / 2)
    dfforce = alfa_cl * v ** 2 * (l - CoPy)
    drforce = alfa_cd * v ** 2 * (CoPz + w / 2)
    cenforce = m_rear * v ** 2 / r

    rearNForce = (massforce + accforce + dfforce + drforce) / l
    normal_force = (rearNForce / 2 + cenforce * (h + w / 2)) / d

    return normal_force


def normal_force_rear_inner(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, v, d, acc):
    """Calculation of normal force on rear inner tire"""

    m_rear = m * a / l
    massforce = m * g * b
    accforce = acc * m * (h + w / 2)
    dfforce = alfa_cl * v ** 2 * (l - CoPy)
    drforce = alfa_cd * v ** 2 * (CoPz + w / 2)
    cenforce = m_rear * v ** 2 / r

    rearNForce = (massforce + accforce + dfforce + drforce) / l
    normal_force = (rearNForce / 2 - cenforce * (h + w / 2)) / d

    return normal_force


def normal_force_front_outer(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, v, d, acc):
    """Calculation of normal force on front outer tire"""

    m_front = m * b / l
    massforce = m * g * b
    accforce = acc * m * (h + w / 2)
    dfforce = alfa_cl * v ** 2 * (l - CoPy)
    drforce = alfa_cd * v ** 2 * (CoPz + w / 2)
    cenforce = m_front * v ** 2 / r

    frontNForce = (massforce - accforce + dfforce - drforce) / l
    normal_force = (frontNForce / 2 + cenforce * (h + w / 2)) / d

    return normal_force


def normal_force_front_inner(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, v, d, acc):
    """Calculation of normal force on front inner tire"""

    m_front = m * b / l
    massforce = m*g*b
    accforce = acc*m*(h+w/2)
    dfforce = alfa_cl*v**2*(l-CoPy)
    drforce = alfa_cd*v**2*(CoPz + w/2)
    cenforce = m_front*v**2/r

    frontNForce = (massforce - accforce + dfforce - drforce)/l
    normal_force = (frontNForce/2 - cenforce*(h + w/2))/d

    return normal_force


def add_new_points(name_of_track):
    """With approximation add extra points in track"""

    df = pd.read_csv(name_of_track, index_col=0)

    x = df["x"].to_numpy()
    y = df["y"].to_numpy()

    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=2
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = splprep([x, y], s=4)  # parameter s is important for smoothing

    # evaluate the spline fits for (as many as you like) evenly spaced distance values
    xi, yi = splev(np.linspace(0, 1, 10000), tck)

    new_df = pd.DataFrame()
    new_df["x"] = xi
    new_df["y"] = yi

    new_df.to_csv(name_of_track)


def radius_of_corner(name_of_track):
    """From three points of track we get a radius
    apply this for each point of the track"""

    df = pd.read_csv(name_of_track, index_col=0)

    for x in range(2, len(df.index) - 2):
        p1 = np.array([df.loc[x - 2, 'x'], df.loc[x - 2, 'y']])
        p2 = np.array([df.loc[x - 1, 'x'], df.loc[x - 1, 'y']])
        p3 = np.array([df.loc[x, 'x'], df.loc[x, 'y']])
        p4 = np.array([df.loc[x + 1, 'x'], df.loc[x + 1, 'y']])
        p5 = np.array([df.loc[x + 2, 'x'], df.loc[x + 2, 'y']])

        def radij(A, B, C):

            AB = B - A
            BC = C - B
            AC = C - A

            a = np.linalg.norm(AB)
            b = np.linalg.norm(BC)
            c = np.linalg.norm(AC)
            d = np.sqrt(2.0 * a ** 2 * b ** 2 +
                    2.0 * b ** 2 * c ** 2 +
                    2.0 * c ** 2 * a ** 2 -
                    a ** 4 - b ** 4 - c ** 4)

            if d == 0:
                # if their is no difference between this and previous point, radius is given as 0
                r = 150
                return r
            else:
                r = (a * b * c) / d
                return r

        radius = sum([radij(p1, p3, p5), radij(p2, p3, p4), radij(p1, p2, p3), radij(p3, p4, p5)])/4

        t1 = np.array([df.loc[x, 'x'], df.loc[x, 'y']])
        t2 = np.array([df.loc[x + 1, 'x'], df.loc[x + 1, 'y']])
        s = np.sqrt((t2[0] - t1[0]) ** 2 + (t2[1] - t1[1]) ** 2)  # calculation of total distance in x and y direction
        df.at[x, "s"] = s

        df.at[x, "R"] = radius

    df.fillna(method="ffill",
              inplace=True)  # if error occurs, pandas gives NaN value. Automatic filled with previous value

    df.to_csv(name_of_track)


def max_velocity_front_inner(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, mu, K, d):

    m_rear = m * b / l
    m_front = m * a / l
    x = 0

    c4 = 0.0625*(4.0*CoPy**2*alfa_cl**2*d**2*mu**2*r**2 + 8.0*CoPy*CoPz*alfa_cd*alfa_cl*d**2*mu**2*r**2 +
                 4.0*CoPy*alfa_cd*alfa_cl*d**2*mu**2*r**2*w - 8.0*CoPy*alfa_cl**2*d**2*l*mu**2*r**2 +
                 16.0*CoPy*alfa_cl*d*h*l*m_front*mu**2*r + 8.0*CoPy*alfa_cl*d*l*m_front*mu**2*r*w +
                 4.0*CoPz**2*alfa_cd**2*d**2*mu**2*r**2 + 4.0*CoPz*alfa_cd**2*d**2*mu**2*r**2*w -
                 8.0*CoPz*alfa_cd*alfa_cl*d**2*l*mu**2*r**2 + 16.0*CoPz*alfa_cd*d*h*l*m_front*mu**2*r +
                 8.0*CoPz*alfa_cd*d*l*m_front*mu**2*r*w - 16.0*K**2*alfa_cd**2*d**2*l**2*r**2 +
                 1.0*alfa_cd**2*d**2*mu**2*r**2*w**2 - 4.0*alfa_cd*alfa_cl*d**2*l*mu**2*r**2*w +
                 8.0*alfa_cd*d*h*l*m_front*mu**2*r*w + 4.0*alfa_cd*d*l*m_front*mu**2*r*w**2 +
                 4.0*alfa_cl**2*d**2*l**2*mu**2*r**2 - 16.0*alfa_cl*d*h*l**2*m_front*mu**2*r -
                 8.0*alfa_cl*d*l**2*m_front*mu**2*r*w - 4.0*d**2*l**2*m_front**2 + 16.0*h**2*l**2*m_front**2*mu**2 +
                 16.0*h*l**2*m_front**2*mu**2*w + 4.0*l**2*m_front**2*mu**2*w**2)/(d**2*l**2*r**2)

    c2 = 0.125*(-4.0*CoPy*alfa_cl*b*d*g*m*mu**2*r + 4.0*CoPy*alfa_cl*d*h*m*mu**2*r*x + 2.0*CoPy*alfa_cl*d*m*mu**2*r*w*x
                - 4.0*CoPz*alfa_cd*b*d*g*m*mu**2*r + 4.0*CoPz*alfa_cd*d*h*m*mu**2*r*x + 2.0*CoPz*alfa_cd*d*m*mu**2*r*w*x
                - 2.0*alfa_cd*b*d*g*m*mu**2*r*w + 2.0*alfa_cd*d*h*m*mu**2*r*w*x + 1.0*alfa_cd*d*m*mu**2*r*w**2*x +
                4.0*alfa_cl*b*d*g*l*m*mu**2*r - 4.0*alfa_cl*d*h*l*m*mu**2*r*x - 2.0*alfa_cl*d*l*m*mu**2*r*w*x -
                8.0*b*g*h*l*m*m_front*mu**2 - 4.0*b*g*l*m*m_front*mu**2*w + 8.0*h**2*l*m*m_front*mu**2*x +
                8.0*h*l*m*m_front*mu**2*w*x + 2.0*l*m*m_front*mu**2*w**2*x)/(d*l**2*r)

    c0 = 0.0625*(4.0*b**2*g**2*m**2*mu**2 - 8.0*b*g*h*m**2*mu**2*x - 4.0*b*g*m**2*mu**2*w*x + 4.0*h**2*m**2*mu**2*x**2
                 + 4.0*h*m**2*mu**2*w*x**2 + 1.0*m**2*mu**2*w**2*x**2)/l**2

    coeff = [c4, 0, c2, 0, c0]

    return coeff


def max_velocity_front_outer(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, mu, K, d):

    m_rear = m * b / l
    m_front = m * a / l
    x = 0

    c4 = 0.0625*(4.0*CoPy**2*alfa_cl**2*d**2*mu**2*r**2 + 8.0*CoPy*CoPz*alfa_cd*alfa_cl*d**2*mu**2*r**2 +
                 4.0*CoPy*alfa_cd*alfa_cl*d**2*mu**2*r**2*w - 8.0*CoPy*alfa_cl**2*d**2*l*mu**2*r**2 -
                 16.0*CoPy*alfa_cl*d*h*l*m_front*mu**2*r - 8.0*CoPy*alfa_cl*d*l*m_front*mu**2*r*w +
                 4.0*CoPz**2*alfa_cd**2*d**2*mu**2*r**2 + 4.0*CoPz*alfa_cd**2*d**2*mu**2*r**2*w -
                 8.0*CoPz*alfa_cd*alfa_cl*d**2*l*mu**2*r**2 - 16.0*CoPz*alfa_cd*d*h*l*m_front*mu**2*r -
                 8.0*CoPz*alfa_cd*d*l*m_front*mu**2*r*w - 16.0*K**2*alfa_cd**2*d**2*l**2*r**2 +
                 1.0*alfa_cd**2*d**2*mu**2*r**2*w**2 - 4.0*alfa_cd*alfa_cl*d**2*l*mu**2*r**2*w -
                 8.0*alfa_cd*d*h*l*m_front*mu**2*r*w - 4.0*alfa_cd*d*l*m_front*mu**2*r*w**2 +
                 4.0*alfa_cl**2*d**2*l**2*mu**2*r**2 + 16.0*alfa_cl*d*h*l**2*m_front*mu**2*r +
                 8.0*alfa_cl*d*l**2*m_front*mu**2*r*w - 4.0*d**2*l**2*m_front**2 + 16.0*h**2*l**2*m_front**2*mu**2 +
                 16.0*h*l**2*m_front**2*mu**2*w + 4.0*l**2*m_front**2*mu**2*w**2)/(d**2*l**2*r**2)

    c2 = 0.125*(-4.0*CoPy*alfa_cl*b*d*g*m*mu**2*r + 4.0*CoPy*alfa_cl*d*h*m*mu**2*r*x + 2.0*CoPy*alfa_cl*d*m*mu**2*r*w*x
                - 4.0*CoPz*alfa_cd*b*d*g*m*mu**2*r + 4.0*CoPz*alfa_cd*d*h*m*mu**2*r*x + 2.0*CoPz*alfa_cd*d*m*mu**2*r*w*x
                - 2.0*alfa_cd*b*d*g*m*mu**2*r*w + 2.0*alfa_cd*d*h*m*mu**2*r*w*x + 1.0*alfa_cd*d*m*mu**2*r*w**2*x +
                4.0*alfa_cl*b*d*g*l*m*mu**2*r - 4.0*alfa_cl*d*h*l*m*mu**2*r*x - 2.0*alfa_cl*d*l*m*mu**2*r*w*x +
                8.0*b*g*h*l*m*m_front*mu**2 + 4.0*b*g*l*m*m_front*mu**2*w - 8.0*h**2*l*m*m_front*mu**2*x -
                8.0*h*l*m*m_front*mu**2*w*x - 2.0*l*m*m_front*mu**2*w**2*x)/(d*l**2*r)


    c0 = 0.0625*(4.0*b**2*g**2*m**2*mu**2 - 8.0*b*g*h*m**2*mu**2*x - 4.0*b*g*m**2*mu**2*w*x + 4.0*h**2*m**2*mu**2*x**2
                 + 4.0*h*m**2*mu**2*w*x**2 + 1.0*m**2*mu**2*w**2*x**2)/l**2

    coeff = [c4, 0, c2, 0, c0]

    return coeff


def max_velocity_rear_outer(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, mu, K, d):

    m_rear = m * b / l
    m_front = m * a / l
    x = 0

    c4 = 0.0625*(4.0*CoPy**2*alfa_cl**2*d**2*mu**2*r**2 - 8.0*CoPy*CoPz*alfa_cd*alfa_cl*d**2*mu**2*r**2 -
                 4.0*CoPy*alfa_cd*alfa_cl*d**2*mu**2*r**2*w - 8.0*CoPy*alfa_cl**2*d**2*l*mu**2*r**2 -
                 16.0*CoPy*alfa_cl*d*h*l*m_rear*mu**2*r - 8.0*CoPy*alfa_cl*d*l*m_rear*mu**2*r*w +
                 4.0*CoPz**2*alfa_cd**2*d**2*mu**2*r**2 + 4.0*CoPz*alfa_cd**2*d**2*mu**2*r**2*w +
                 8.0*CoPz*alfa_cd*alfa_cl*d**2*l*mu**2*r**2 + 16.0*CoPz*alfa_cd*d*h*l*m_rear*mu**2*r +
                 8.0*CoPz*alfa_cd*d*l*m_rear*mu**2*r*w - 16.0*K**2*alfa_cd**2*d**2*l**2*r**2 +
                 1.0*alfa_cd**2*d**2*mu**2*r**2*w**2 + 4.0*alfa_cd*alfa_cl*d**2*l*mu**2*r**2*w +
                 8.0*alfa_cd*d*h*l*m_rear*mu**2*r*w + 4.0*alfa_cd*d*l*m_rear*mu**2*r*w**2 +
                 4.0*alfa_cl**2*d**2*l**2*mu**2*r**2 + 16.0*alfa_cl*d*h*l**2*m_rear*mu**2*r +
                 8.0*alfa_cl*d*l**2*m_rear*mu**2*r*w - 4.0*d**2*l**2*m_rear**2 + 16.0*h**2*l**2*m_rear**2*mu**2 +
                 16.0*h*l**2*m_rear**2*mu**2*w + 4.0*l**2*m_rear**2*mu**2*w**2)/(d**2*l**2*r**2)

    c2 = 0.125*(-4.0*CoPy*a*alfa_cl*d*g*m*mu**2*r - 4.0*CoPy*alfa_cl*d*h*m*mu**2*r*x - 2.0*CoPy*alfa_cl*d*m*mu**2*r*w*x
                + 4.0*CoPz*a*alfa_cd*d*g*m*mu**2*r + 4.0*CoPz*alfa_cd*d*h*m*mu**2*r*x + 2.0*CoPz*alfa_cd*d*m*mu**2*r*w*x
                + 2.0*a*alfa_cd*d*g*m*mu**2*r*w + 4.0*a*alfa_cl*d*g*l*m*mu**2*r + 8.0*a*g*h*l*m*m_rear*mu**2 +
                4.0*a*g*l*m*m_rear*mu**2*w + 2.0*alfa_cd*d*h*m*mu**2*r*w*x + 1.0*alfa_cd*d*m*mu**2*r*w**2*x +
                4.0*alfa_cl*d*h*l*m*mu**2*r*x + 2.0*alfa_cl*d*l*m*mu**2*r*w*x + 8.0*h**2*l*m*m_rear*mu**2*x +
                8.0*h*l*m*m_rear*mu**2*w*x + 2.0*l*m*m_rear*mu**2*w**2*x)/(d*l**2*r)

    c0 = 0.0625*(4.0*a**2*g**2*m**2*mu**2 + 8.0*a*g*h*m**2*mu**2*x + 4.0*a*g*m**2*mu**2*w*x + 4.0*h**2*m**2*mu**2*x**2
                 + 4.0*h*m**2*mu**2*w*x**2 + 1.0*m**2*mu**2*w**2*x**2)/l**2

    coeff = [c4, 0, c2, 0, c0]

    return coeff


def max_velocity_rear_inner(a, b, m, g, h, w, alfa_cl, l, CoPy, alfa_cd, CoPz, r, mu, K, d):

    m_rear = m * b / l
    m_front = m * a / l
    x = 0

    c4 = 0.0625*(4.0*CoPy**2*alfa_cl**2*d**2*mu**2*r**2 + 8.0*CoPy*CoPz*alfa_cd*alfa_cl*d**2*mu**2*r**2 +
                 4.0*CoPy*alfa_cd*alfa_cl*d**2*mu**2*r**2*w - 8.0*CoPy*alfa_cl**2*d**2*l*mu**2*r**2 +
                 16.0*CoPy*alfa_cl*d*h*l*m_rear*mu**2*r + 8.0*CoPy*alfa_cl*d*l*m_rear*mu**2*r*w +
                 4.0*CoPz**2*alfa_cd**2*d**2*mu**2*r**2 + 4.0*CoPz*alfa_cd**2*d**2*mu**2*r**2*w -
                 8.0*CoPz*alfa_cd*alfa_cl*d**2*l*mu**2*r**2 + 16.0*CoPz*alfa_cd*d*h*l*m_rear*mu**2*r +
                 8.0*CoPz*alfa_cd*d*l*m_rear*mu**2*r*w - 16.0*K**2*alfa_cd**2*d**2*l**2*r**2 +
                 1.0*alfa_cd**2*d**2*mu**2*r**2*w**2 - 4.0*alfa_cd*alfa_cl*d**2*l*mu**2*r**2*w +
                 8.0*alfa_cd*d*h*l*m_rear*mu**2*r*w + 4.0*alfa_cd*d*l*m_rear*mu**2*r*w**2 +
                 4.0*alfa_cl**2*d**2*l**2*mu**2*r**2 - 16.0*alfa_cl*d*h*l**2*m_rear*mu**2*r -
                 8.0*alfa_cl*d*l**2*m_rear*mu**2*r*w - 4.0*d**2*l**2*m_rear**2 + 16.0*h**2*l**2*m_rear**2*mu**2 +
                 16.0*h*l**2*m_rear**2*mu**2*w + 4.0*l**2*m_rear**2*mu**2*w**2)/(d**2*l**2*r**2)

    c2 = 0.125*(-4.0*CoPy*a*alfa_cl*d*g*m*mu**2*r - 4.0*CoPy*alfa_cl*d*h*m*mu**2*r*x - 2.0*CoPy*alfa_cl*d*m*mu**2*r*w*x
                - 4.0*CoPz*a*alfa_cd*d*g*m*mu**2*r - 4.0*CoPz*alfa_cd*d*h*m*mu**2*r*x - 2.0*CoPz*alfa_cd*d*m*mu**2*r*w*x
                - 2.0*a*alfa_cd*d*g*m*mu**2*r*w + 4.0*a*alfa_cl*d*g*l*m*mu**2*r - 8.0*a*g*h*l*m*m_rear*mu**2 -
                4.0*a*g*l*m*m_rear*mu**2*w - 2.0*alfa_cd*d*h*m*mu**2*r*w*x - 1.0*alfa_cd*d*m*mu**2*r*w**2*x +
                4.0*alfa_cl*d*h*l*m*mu**2*r*x + 2.0*alfa_cl*d*l*m*mu**2*r*w*x - 8.0*h**2*l*m*m_rear*mu**2*x -
                8.0*h*l*m*m_rear*mu**2*w*x - 2.0*l*m*m_rear*mu**2*w**2*x)/(d*l**2*r)

    c0 = 0.0625*(4.0*a**2*g**2*m**2*mu**2 + 8.0*a*g*h*m**2*mu**2*x + 4.0*a*g*m**2*mu**2*w*x + 4.0*h**2*m**2*mu**2*x**2
                 + 4.0*h*m**2*mu**2*w*x**2 + 1.0*m**2*mu**2*w**2*x**2)/l**2

    coeff = [c4, 0, c2, 0, c0]

    return coeff

def skidpad_time(track_width, a, b, mass, height_CG, w, alpha_Cl, wheelbase, CoPy, alpha_Cd, CoPz, coef_friction,
                 KF, KR, v_max_teo):
    r = 15.25/2 + 0.1 + track_width/2

    max_vel_fi = max_velocity_front_inner(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase,
                                          CoPy=CoPy,
                                          alfa_cd=alpha_Cd, CoPz=CoPz, r=r, mu=coef_friction, K=KF,
                                          d=track_width)

    max_vel_ri = max_velocity_rear_inner(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase,
                                         CoPy=CoPy,
                                         alfa_cd=alpha_Cd, CoPz=CoPz, r=r, mu=coef_friction, K=KR,
                                         d=track_width)

    max_vel_fo = max_velocity_front_outer(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase,
                                          CoPy=CoPy,
                                          alfa_cd=alpha_Cd, CoPz=CoPz, r=r, mu=coef_friction, K=KR,
                                          d=track_width)

    max_vel_ro = max_velocity_rear_outer(a, b, m=mass, g=g, h=height_CG, w=w, alfa_cl=alpha_Cl, l=wheelbase,
                                         CoPy=CoPy,
                                         alfa_cd=alpha_Cd, CoPz=CoPz, r=r, mu=coef_friction, K=KR,
                                         d=track_width)

    def vel_fi(x):
        c4 = max_vel_fi[0]
        c2 = max_vel_fi[2]
        c0 = max_vel_fi[4]
        return (c4 * x ** 4 + c2 * x ** 2 + c0)

    def vel_fo(x):
        c4 = max_vel_fo[0]
        c2 = max_vel_fo[2]
        c0 = max_vel_fo[4]
        return (c4 * x ** 4 + c2 * x ** 2 + c0)

    def vel_ri(x):
        c4 = max_vel_ri[0]
        c2 = max_vel_ri[2]
        c0 = max_vel_ri[4]
        return (c4 * x ** 4 + c2 * x ** 2 + c0)

    def vel_ro(x):
        c4 = max_vel_ro[0]
        c2 = max_vel_ro[2]
        c0 = max_vel_ro[4]
        return (c4 * x ** 4 + c2 * x ** 2 + c0)

    root_fi = optimize.newton(vel_fi, v_max_teo, maxiter=100000)
    root_ri = optimize.newton(vel_ri, v_max_teo, maxiter=100000)
    root_fo = optimize.newton(vel_fo, v_max_teo, maxiter=100000)
    root_ro = optimize.newton(vel_ro, v_max_teo, maxiter=100000)

    velocity = min((root_fi+root_fo)/2, (root_ri+root_ro)/2)

    time_skidpad = 2*np.pi*r/velocity

    return time_skidpad

def max_vel_rear_inner_v2(track_width, a, mass, height_CG, alpha_Cl, wheelbase, CoPy, alpha_Cd, CoPz, coef_friction,
                          v_max_teo, tire_radius, r, mass_rear):

    def forces_calc(v):
        mass_part = mass * g * a
        downforce_part = alpha_Cl * CoPy
        drag_part = alpha_Cd * (CoPz + tire_radius)
        centripental_part = mass_rear / r * (height_CG + tire_radius)

        return ((coef_friction * ((track_width / 2 / wheelbase * (
                    mass_part + downforce_part * v ** 2 + drag_part * v ** 2) - centripental_part * v ** 2) / wheelbase))) ** (
                           1 / 2) - drag_part * v ** 2

    if r < 10:
        root = optimize.newton(forces_calc, 5, maxiter=100000)

    elif 10 <= r < 50:
        root = optimize.newton(forces_calc, 10, maxiter=100000)

    else:
        root = optimize.newton(forces_calc, v_max_teo, maxiter=100000)

    return root.real


def max_vel_front_inner_v2(track_width, b, mass, height_CG, alpha_Cl, wheelbase, CoPy, alpha_Cd, CoPz,
                          coef_friction, v_max_teo, tire_radius, r, mass_front):
    def forces_calc(v):
        mass_part = mass * g * b
        downforce_part = alpha_Cl * (wheelbase - CoPy)
        drag_part = alpha_Cd * (CoPz + tire_radius)
        centripental_part = mass_front / r * (height_CG + tire_radius)

        return (((mass_part + downforce_part * v ** 2 - drag_part * v ** 2) * track_width / 2 - centripental_part * v ** 2) * coef_friction / track_width) ** 2 - (
                           centripental_part * v ** 2) ** 2

    if r < 10:
        root = optimize.newton(forces_calc, 10, maxiter=100000)

    elif 10 <= r < 50:
        root = optimize.newton(forces_calc, 20, maxiter=100000)

    else:
        root = optimize.newton(forces_calc, v_max_teo, maxiter=100000)

    return root.real
