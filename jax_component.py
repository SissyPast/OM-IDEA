import jax.numpy as jnp


def extrapolate(x, xp, yp):
    """np.interp function with linear extrapolation"""
    y = jnp.interp(x, xp, yp)
    y = jnp.where(
        x < xp[0],
        yp[0] + (x - xp[0]) * (yp[0] - yp[1]) / (xp[0] - xp[1]),
        y,
    )
    y = jnp.where(
        x > xp[-1],
        yp[-1] + (x - xp[-1]) * (yp[-1] - yp[-2]) / (xp[-1] - xp[-2]),
        y,
    )
    return y
