import numpy as np
import matplotlib.pyplot as plt
from app.utils.mittag_leffler import ml


def model_jonscher_url(omega, m, n):
    omega_m = np.power(omega, m)
    omega_n_minus_1 = np.power(omega, n - 1)
    return omega_m, omega_n_minus_1


def susceptibility_cc(omega, tau_star, alpha):
    chi = np.divide(1, np.add(1, np.power(np.multiply(omega, complex(0, tau_star)), alpha)))
    return np.real(chi), np.imag(chi), np.subtract(1, np.real(chi))


def susceptibility_hn(omega, tau_star, alpha, beta):
    chi = np.divide(
        1,
        np.power(
            np.add(
                1,
                np.power(
                    np.multiply(omega, complex(0, tau_star)),
                    alpha
                )
            ),
            beta
        )
    )
    return np.real(chi), np.imag(chi), np.subtract(1, np.real(chi))


def response_cc(time, tau_star, alpha):
    phi = np.multiply(
        np.multiply(np.divide(1, tau_star),
                    np.power(np.divide(time, tau_star), alpha - 1)),
        ml(-np.power(np.divide(time, tau_star), alpha),
           alpha,
           alpha,
           1)
    )
    return phi


def response_hn(time, tau_star, alpha, beta):
    phi = np.multiply(
        np.multiply(
            np.divide(1, tau_star),
            np.power(np.divide(time, tau_star), alpha * beta - 1)
        ),
        ml(-np.power(np.divide(time, tau_star), alpha),
           alpha,
           alpha * beta,
           beta)
    )
    return phi


def relaxation_cc(time, tau_star, alpha):
    psi = ml(-np.power(np.divide(time, tau_star), alpha),
             alpha,
             1,
             1)
    return psi


def relaxation_hn(time, tau_star, alpha, beta):
    psi = np.subtract(
        1,
        np.multiply(
            np.power(np.divide(time, tau_star), alpha * beta),
            ml(-np.power(np.divide(time, tau_star), alpha),
               alpha,
               alpha * beta + 1,
               beta)
        )
    )
    return psi


def lin_time_spectral_distribution_cc(tau, tau_star, alpha):
    h = np.multiply(
        1 / (np.pi * tau_star),
        np.divide(
            np.multiply(np.sin(alpha * np.pi), np.power(np.divide(tau, tau_star), alpha - 1)),
            np.add(
                np.add(
                    np.power(np.divide(tau, tau_star), 2 * alpha),
                    np.multiply(np.power(np.divide(tau, tau_star), alpha), 2 * np.cos(alpha * np.pi))
                ),
                1
            )
        )
    )
    return h


def lin_time_spectral_distribution_hn(tau, tau_star, alpha, beta):
    # theta = np.arctan(np.divide(
    #     np.multiply(
    #         np.power(
    #             np.divide(tau_star, tau),
    #             alpha
    #         ),
    #         np.sin(alpha * np.pi)
    #     ),
    #     np.add(
    #         1,
    #         np.multiply(
    #             np.power(
    #                 np.divide(tau_star, tau),
    #                 alpha
    #             ),
    #             np.cos(alpha * np.pi)
    #         )
    #     )
    # ))
    theta = np.subtract(
        np.pi / 2,
        np.arctan(np.divide(
            np.add(
                1,
                np.multiply(
                    np.power(
                        np.divide(tau_star, tau),
                        alpha
                    ),
                    np.cos(alpha * np.pi)
                )
            ),
            np.multiply(
                np.power(
                    np.divide(tau_star, tau),
                    alpha
                ),
                np.sin(alpha * np.pi)
            )
        ))
    )
    h = np.multiply(
        np.divide(
            1,
            np.multiply(np.pi, tau)
        ),
        np.divide(
            np.sin(np.multiply(
                beta,
                theta
            )),
            np.power(
                np.add(
                    1,
                    np.add(
                        np.multiply(
                            np.power(
                                np.divide(tau_star, tau),
                                alpha
                            ),
                            2 * np.cos(alpha * np.pi)
                        ),
                        np.power(
                            np.divide(tau_star, tau),
                            2 * alpha
                        )
                    )
                ),
                beta / 2
            )
        )
    )
    return h


def log_time_spectral_distribution_cc(u, tau_star, alpha):
    l = np.multiply(
        1 / (2 * np.pi),
        np.divide(
            np.sin(alpha * np.pi),
            np.add(
                np.cosh(np.multiply(alpha, np.subtract(u, np.log(tau_star)))),
                np.cos(alpha * np.pi)
            )
        )
    )
    return l


def log_time_spectral_distribution_hn(u, tau_star, alpha, beta):
    # !!!This is the wrong theta:
    # theta = np.arctan(np.divide(
    #                 np.multiply(
    #                     np.power(
    #                         np.divide(tau_star, np.exp(u)),
    #                         alpha
    #                     ),
    #                     np.sin(alpha * np.pi)
    #                 ),
    #                 np.add(
    #                     1,
    #                     np.multiply(
    #                         np.power(
    #                             np.divide(tau_star, np.exp(u)),
    #                             alpha
    #                         ),
    #                         np.cos(alpha * np.pi)
    #                     )
    #                 )
    #             ))
    theta = np.subtract(
        np.pi / 2,
        np.arctan(np.divide(
            np.add(
                1,
                np.multiply(
                    np.power(
                        np.divide(tau_star, np.exp(u)),
                        alpha
                    ),
                    np.cos(alpha * np.pi)
                )
            ),
            np.multiply(
                np.power(
                    np.divide(tau_star, np.exp(u)),
                    alpha
                ),
                np.sin(alpha * np.pi)
            )
        ))
    )
    l = np.multiply(
        1 / (np.pi),
        np.divide(
            np.sin(np.multiply(
                beta,
                theta)
            ),
            np.power(
                np.add(
                    1,
                    np.add(
                        np.multiply(
                            np.power(
                                np.divide(tau_star, np.exp(u)),
                                alpha
                            ),
                            2 * np.cos(alpha * np.pi)
                        ),
                        np.power(
                            np.divide(tau_star, np.exp(u)),
                            2 * alpha
                        )
                    )
                ),
                beta / 2
            )
        )
    )
    return l


if __name__ == "__main__":
    omega = np.power(10, np.arange(-4, 4, 0.01))
    time_lin = np.arange(0.01, 2.0, 0.01)
    time_log = np.power(10, np.arange(-2, 2, 0.01))
    tau_lin = np.arange(0.01, 3, 0.01)
    u_lin = np.arange(-3, 3, 0.01)

    fig = plt.figure()
    plt.ion()
    plt.show()
    ax_1 = fig.add_subplot(3, 2, 1)
    ax_1.set_title('Susceptibility')
    ax_1.set_xscale('log')
    ax_1.set_yscale('log')
    ax_2 = fig.add_subplot(3, 2, 2)
    ax_3 = fig.add_subplot(3, 2, 3)
    ax_4 = fig.add_subplot(3, 2, 4)
    ax_2.set_title('Cole-Cole Plot')
    ax_3.set_title('Response Function')
    ax_4.set_title('Relaxation Function')
    ax_5 = fig.add_subplot(3, 2, 5)
    ax_6 = fig.add_subplot(3, 2, 6)
    ax_5.set_title('LIN Time Spectral Distribution of Relaxation Time')
    ax_6.set_title('LOG Time Spectral Distribution of Relaxation Time')

    # 1--Cole-Cole
    # 2--Dividson-Cole
    # 3--Havriliak-Negami beta-fixed
    # 4--Havriliak-Negami alpha-fixed

    model_num = 4

    alpha = 0.6
    beta = 0.8
    alpha_list = np.arange(0.5, 1.0, 0.1)
    beta_list = np.arange(0.5, 1.0, 0.1)

    omega_m = np.array(np.shape(omega))
    omega_n_minus_1 = np.array(np.shape(omega))
    chi_real = np.array(np.shape(omega))
    chi_imag = np.array(np.shape(omega))
    chi_0_minus_chi_real = np.array(np.shape(omega))
    if model_num == 1:
        omega_m, omega_n_minus_1 = model_jonscher_url(omega, m=alpha, n=1 - alpha)
        chi_real, chi_imag, chi_0_minus_chi_real = susceptibility_cc(omega, tau_star=1, alpha=alpha)
    elif model_num == 3 or model_num == 4:
        omega_m, omega_n_minus_1 = model_jonscher_url(omega, m=alpha, n=1 - alpha * beta)
        chi_real, chi_imag, chi_0_minus_chi_real = susceptibility_hn(omega, tau_star=1, alpha=alpha, beta=beta)
    ax_1.plot(omega, omega_m, color='r', alpha=1, linestyle='--', label='omega^m')
    ax_1.plot(omega, omega_n_minus_1, color='b', alpha=1, linestyle='--', label='omega^(n-1)')
    ax_1.plot(omega, chi_real, color='black', alpha=1, linestyle='-', label='chi_real')
    ax_1.plot(omega, -chi_imag, color='gray', alpha=1, linestyle='-', label='chi_imag')
    ax_1.plot(omega, chi_0_minus_chi_real, color='black', alpha=1, linestyle='-', label='chi_0-_chi_real')

    phi = np.array(np.shape(time_lin))
    psi = np.array(np.shape(time_log))
    h = np.array(np.shape(tau_lin))
    l = np.array(np.shape(u_lin))
    if model_num == 1 or model_num == 3:
        for alpha_ in alpha_list:
            if model_num == 1:
                chi_real, chi_imag, chi_0_minus_chi_real = susceptibility_cc(omega, tau_star=1, alpha=alpha_)
                phi = response_cc(time_lin, tau_star=1, alpha=alpha_)
                psi = relaxation_cc(time_log, tau_star=1, alpha=alpha_)
                h = lin_time_spectral_distribution_cc(tau=tau_lin, tau_star=1, alpha=alpha_)
                l = log_time_spectral_distribution_cc(u=u_lin, tau_star=1, alpha=alpha_)
            elif model_num == 3:
                chi_real, chi_imag, chi_0_minus_chi_real = susceptibility_hn(omega, tau_star=1, alpha=alpha_, beta=beta)
                phi = response_hn(time_lin, tau_star=1, alpha=alpha_, beta=beta)
                psi = relaxation_hn(time_log, tau_star=1, alpha=alpha_, beta=beta)
                h = lin_time_spectral_distribution_hn(tau=tau_lin, tau_star=1, alpha=alpha_, beta=beta)
                l = log_time_spectral_distribution_hn(u=u_lin, tau_star=1, alpha=alpha_, beta=beta)
            ax_2.plot(chi_real, -chi_imag, color=str(1 - alpha_), alpha=1, linestyle='-', label='alpha=' + str(alpha_))
            ax_3.plot(time_lin, phi, color=str(1 - alpha_), alpha=1, linestyle='-', label='phi,alpha=' + str(alpha_))
            ax_3.set_ylim([0, 2])
            ax_4.set_xscale('log')
            ax_4.plot(time_log, psi, color=str(1 - alpha_), alpha=1, linestyle='-', label='psi,alpha=' + str(alpha_))
            ax_5.plot(tau_lin, h, color=str(1 - alpha_), alpha=1, linestyle='-', label='H,alpha=' + str(alpha_))
            ax_6.plot(u_lin, l, color=str(1 - alpha_), alpha=1, linestyle='-', label='L,alpha=' + str(alpha_))
    elif model_num == 4:
        for beta_ in beta_list:
            chi_real, chi_imag, chi_0_minus_chi_real = susceptibility_hn(omega, tau_star=1, alpha=alpha, beta=beta_)
            phi = response_hn(time_lin, tau_star=1, alpha=alpha, beta=beta_)
            psi = relaxation_hn(time_log, tau_star=1, alpha=alpha, beta=beta_)
            h = lin_time_spectral_distribution_hn(tau=tau_lin, tau_star=1, alpha=alpha, beta=beta_)
            l = log_time_spectral_distribution_hn(u=u_lin, tau_star=1, alpha=alpha, beta=beta_)

            ax_2.plot(chi_real, -chi_imag, color=str(1 - beta_), alpha=1, linestyle='-', label='beta=' + str(beta_))
            ax_3.plot(time_lin, phi, color=str(1 - beta_), alpha=1, linestyle='-', label='phi,beta=' + str(beta_))
            ax_3.set_ylim([0, 2])
            ax_4.set_xscale('log')
            ax_4.plot(time_log, psi, color=str(1 - beta_), alpha=1, linestyle='-', label='psi,beta=' + str(beta_))
            ax_5.plot(tau_lin, h, color=str(1 - beta_), alpha=1, linestyle='-', label='H,beta=' + str(beta_))
            ax_6.plot(u_lin, l, color=str(1 - beta_), alpha=1, linestyle='-', label='L,beta=' + str(beta_))

    ax_1.legend()
    ax_2.legend()
    ax_3.legend()
    ax_4.legend()
    ax_5.legend()
    ax_6.legend()
