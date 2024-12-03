macro_rules! to_degrees_impl {
    ($v:expr) => {
        AutoFloat {
            x: $v.x.to_degrees(),
            dx: unary_op($v.dx, |v| v.to_degrees()),
        }
    };
}
pub(crate) use to_degrees_impl;

macro_rules! to_radians_impl {
    ($v:expr) => {
        AutoFloat {
            x: $v.x.to_radians(),
            dx: unary_op($v.dx, |v| v.to_radians()),
        }
    };
}
pub(crate) use to_radians_impl;

macro_rules! max_impl {
    ($lhs:expr, $rhs:expr) => {
        if $lhs.x < $rhs.x {
            $rhs
        } else {
            $lhs
        }
    };
}
pub(crate) use max_impl;

macro_rules! min_impl {
    ($lhs:expr, $rhs:expr) => {
        if $lhs.x > $rhs.x {
            $rhs
        } else {
            $lhs
        }
    };
}
pub(crate) use min_impl;

#[cfg(feature = "float_impl")]
mod float_impl {
    macro_rules! floor_impl {
        ($lhs:expr) => {
            AutoFloat::constant($lhs.x.floor())
        };
    }
    pub(crate) use floor_impl;

    macro_rules! ceil_impl {
        ($lhs:expr) => {
            AutoFloat::constant($lhs.x.ceil())
        };
    }
    pub(crate) use ceil_impl;

    macro_rules! round_impl {
        ($lhs:expr) => {
            AutoFloat::constant($lhs.x.round())
        };
    }
    pub(crate) use round_impl;

    macro_rules! trunc_impl {
        ($lhs:expr) => {
            AutoFloat::constant($lhs.x.trunc())
        };
    }
    pub(crate) use trunc_impl;

    macro_rules! fract_impl {
        ($lhs:expr) => {
            AutoFloat {
                x: $lhs.x.fract(),
                dx: $lhs.dx,
            }
        };
    }
    pub(crate) use fract_impl;

    macro_rules! abs_impl {
        ($lhs:expr) => {
            if $lhs.x >= T::zero() {
                $lhs
            } else {
                -$lhs
            }
        };
    }
    pub(crate) use abs_impl;

    macro_rules! mul_add_impl {
        ($a:expr, $b:expr,$c:expr) => {{
            AutoFloat {
                x: $a.x.clone().mul_add($b.x.clone(), $c.x.clone()),
                dx: binary_op(
                    binary_op($a.dx, $b.dx, |l, r| l * $b.x.clone() + r * $a.x.clone()),
                    $c.dx,
                    |l, r| l + r,
                ),
            }
        }};
    }
    pub(crate) use mul_add_impl;

    macro_rules! recip_impl {
        ($lhs:expr) => {{
            let factor = -T::one() / ($lhs.x.clone() * $lhs.x.clone());
            AutoFloat {
                x: $lhs.x.recip(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use recip_impl;

    macro_rules! sin_impl {
        ($lhs:expr) => {{
            let cos_x = $lhs.x.clone().cos();
            AutoFloat {
                x: $lhs.x.sin(),
                dx: unary_op($lhs.dx, |v| v * cos_x.clone()),
            }
        }};
    }
    pub(crate) use sin_impl;

    macro_rules! cos_impl {
        ($lhs:expr) => {{
            let sin_x = -$lhs.x.clone().sin();
            AutoFloat {
                x: $lhs.x.cos(),
                dx: unary_op($lhs.dx, |v| v * sin_x.clone()),
            }
        }};
    }
    pub(crate) use cos_impl;

    macro_rules! tan_impl {
        ($lhs:expr) => {{
            let tan_x = $lhs.x.clone().tan();
            let factor = tan_x.clone() * tan_x.clone() + T::one();
            AutoFloat {
                x: tan_x,
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use tan_impl;

    macro_rules! asin_impl {
        ($lhs:expr) => {{
            let factor = T::one() / (T::one() - $lhs.x.clone() * $lhs.x.clone()).sqrt();
            AutoFloat {
                x: $lhs.x.asin(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use asin_impl;

    macro_rules! acos_impl {
        ($lhs:expr) => {{
            let factor = -T::one() / (T::one() - $lhs.x.clone() * $lhs.x.clone()).sqrt();
            AutoFloat {
                x: $lhs.x.acos(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use acos_impl;

    macro_rules! atan_impl {
        ($lhs:expr) => {{
            let factor = T::one() / ($lhs.x.clone() * $lhs.x.clone() + T::one());
            AutoFloat {
                x: $lhs.x.atan(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use atan_impl;

    macro_rules! atan2_impl {
        ($lhs:expr, $rhs:expr) => {{
            let factor =
                T::one() / ($lhs.x.clone() * $lhs.x.clone() + $rhs.x.clone() * $rhs.x.clone());
            AutoFloat {
                x: $lhs.x.clone().atan2($rhs.x.clone()),
                dx: binary_op($lhs.dx, $rhs.dx, |l, r| {
                    (l * $rhs.x.clone() - r * $lhs.x.clone()) * factor.clone()
                }),
            }
        }};
    }
    pub(crate) use atan2_impl;

    macro_rules! sin_cos_impl {
        ($lhs:expr) => {{
            let (s, c) = $lhs.x.sin_cos();
            let sn = AutoFloat {
                x: s.clone(),
                dx: unary_op($lhs.dx.clone(), |v| v * c.clone()),
            };
            let s_neg = -s;
            let cn = AutoFloat {
                x: c,
                dx: unary_op($lhs.dx, |v| v * s_neg.clone()),
            };
            (sn, cn)
        }};
    }
    pub(crate) use sin_cos_impl;

    macro_rules! sinh_impl {
        ($lhs:expr) => {{
            let cosh_x = $lhs.x.clone().cosh();
            AutoFloat {
                x: $lhs.x.sinh(),
                dx: unary_op($lhs.dx, |v| v * cosh_x.clone()),
            }
        }};
    }
    pub(crate) use sinh_impl;

    macro_rules! cosh_impl {
        ($lhs:expr) => {{
            let sinh_x = $lhs.x.clone().sinh();
            AutoFloat {
                x: $lhs.x.cosh(),
                dx: unary_op($lhs.dx, |v| v * sinh_x.clone()),
            }
        }};
    }
    pub(crate) use cosh_impl;

    macro_rules! tanh_impl {
        ($lhs:expr) => {{
            let tanhx = $lhs.x.clone().tanh();
            let factor = T::one() - tanhx.clone() * tanhx.clone();
            AutoFloat {
                x: tanhx,
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use tanh_impl;

    macro_rules! asinh_impl {
        ($lhs:expr) => {{
            let factor = T::one() / ($lhs.clone().x * $lhs.clone().x + T::one()).sqrt();
            AutoFloat {
                x: $lhs.x.asinh(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use asinh_impl;

    macro_rules! acosh_impl {
        ($lhs:expr) => {{
            let factor = T::one() / ($lhs.x.clone() * $lhs.x.clone() - T::one()).sqrt();
            AutoFloat {
                x: $lhs.x.clone().acosh(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use acosh_impl;

    macro_rules! atanh_impl {
        ($lhs:expr) => {{
            let factor = T::one() / (-$lhs.x.clone() * $lhs.x.clone() + T::one());
            AutoFloat {
                x: $lhs.x.atanh(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use atanh_impl;

    macro_rules! log_impl {
        ($lhs:expr, $rhs:expr) => {{
            let ln_bx = $rhs.x.clone().ln();
            let factor_bdx =
                -$lhs.x.clone().ln() / ($rhs.x.clone() * ln_bx.clone() * ln_bx.clone());
            let factor_sdx = T::one() / ($lhs.x.clone() * ln_bx);

            AutoFloat {
                x: $lhs.x.log($rhs.x),
                dx: binary_op($lhs.dx, $rhs.dx, |l, r| {
                    factor_bdx.clone() * r + factor_sdx.clone() * l
                }),
            }
        }};
    }
    pub(crate) use log_impl;

    macro_rules! log2_impl {
        ($lhs:expr, $two:expr) => {{
            $lhs.log(AutoFloat::constant($two))
        }};
    }
    pub(crate) use log2_impl;

    macro_rules! log10_impl {
        ($lhs:expr, $ten:expr) => {{
            $lhs.log(AutoFloat::constant($ten))
        }};
    }
    pub(crate) use log10_impl;

    macro_rules! ln_impl {
        ($lhs:expr) => {{
            let factor = $lhs.x.clone().recip();
            AutoFloat {
                x: $lhs.x.ln(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use ln_impl;

    macro_rules! ln_1p_impl {
        ($lhs:expr) => {{
            let factor = T::one() / ($lhs.x.clone() + T::one());
            AutoFloat {
                x: $lhs.x.ln_1p(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use ln_1p_impl;

    macro_rules! sqrt_impl {
        ($lhs:expr) => {{
            let denom = $lhs.x.clone().sqrt() * T::from(2).unwrap();
            let factor = if denom.is_zero() {
                T::zero()
            } else {
                T::one() / denom
            };

            AutoFloat {
                x: $lhs.x.sqrt(),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use sqrt_impl;

    macro_rules! cbrt_impl {
        ($lhs:expr) => {{
            let x_cbrt = $lhs.x.cbrt();
            let denom = x_cbrt.clone() * x_cbrt.clone() * T::from(3).unwrap();
            let factor = if denom.is_zero() {
                T::zero()
            } else {
                T::one() / denom
            };

            AutoFloat {
                x: x_cbrt,
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use cbrt_impl;

    macro_rules! exp_impl {
        ($lhs:expr) => {{
            let exp = $lhs.x.exp();
            AutoFloat {
                x: exp.clone(),
                dx: unary_op($lhs.dx, |v| v * exp.clone()),
            }
        }};
    }
    pub(crate) use exp_impl;

    macro_rules! exp2_impl {
        ($lhs:expr) => {{
            let exp2 = $lhs.x.exp2();
            let factor = T::from(2).unwrap().ln() * exp2.clone();
            AutoFloat {
                x: exp2,
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use exp2_impl;

    macro_rules! powi_impl {
        ($lhs:expr, $rhs:expr) => {{
            let factor = $lhs.x.clone().powi($rhs - 1) * T::from($rhs).unwrap();
            AutoFloat {
                x: $lhs.x.powi($rhs),
                dx: unary_op($lhs.dx, |v| v * factor.clone()),
            }
        }};
    }
    pub(crate) use powi_impl;

    macro_rules! powf_impl {
        ($lhs:expr, $rhs:expr, $x:expr) => {{
            // Avoid division by zero.
            let factor = if $lhs.x.clone().is_zero() && $x.is_zero() {
                T::zero()
            } else {
                $x.clone() * $rhs.x.clone() / $lhs.x.clone()
            };
            let lhs_ln = $lhs.x.ln();

            AutoFloat {
                x: $x.clone(),
                dx: binary_op($lhs.dx, $rhs.dx, |l, r| {
                    // Avoid imaginary values in the ln
                    let dn = if r.is_zero() {
                        T::zero()
                    } else {
                        lhs_ln.clone() * r
                    };

                    dn * $x.clone() + l * factor.clone()
                }),
            }
        }};
    }
    pub(crate) use powf_impl;

    macro_rules! exp_m1_impl {
        ($lhs:expr) => {{
            let exp_x = $lhs.x.clone().exp();
            AutoFloat {
                x: $lhs.x.exp_m1(),
                dx: unary_op($lhs.dx, |v| v * exp_x.clone()),
            }
        }};
    }
    pub(crate) use exp_m1_impl;
}

#[cfg(feature = "float_impl")]
pub(crate) use float_impl::*;
