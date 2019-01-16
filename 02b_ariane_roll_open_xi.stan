
functions {
  vector gp_pred_rng(real[] x2,
                     vector z1, real[] x1,
                     real alpha, real rho, real delta) {
    int N1 = rows(z1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] K =   cov_exp_quad(x1, alpha, rho) + diag_matrix(rep_vector(1e-06, N1));
      matrix[N1, N1] L_K = cholesky_decompose(K);

      vector[N1] L_K_div_z1 = mdivide_left_tri_low(L_K, z1);
      vector[N1] K_div_z1 = mdivide_right_tri_low(L_K_div_z1', L_K)';
      matrix[N1, N2] k_x1_x2 = cov_exp_quad(x1, x2, alpha, rho);
      vector[N2] f2_mu = (k_x1_x2' * K_div_z1);
      matrix[N1, N2] v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      matrix[N2, N2] cov_f2 =   cov_exp_quad(x2, alpha, rho) - v_pred' * v_pred
                              + diag_matrix(rep_vector(delta, N2));
      f2 = multi_normal_rng(f2_mu, cov_f2);
    }
    return f2;
  }
}

data {
  int<lower=1> N;

  real x[N];
  vector[N] y;

  int<lower=1> N_predict;
  real x_predict[N_predict];
}

transformed data {
  matrix[N, 4] noise_matrix;
  vector[N] x_vec;

  matrix[N_predict, 4] noise_matrix_predict;
  vector[N_predict] x_vec_predict;
  
  noise_matrix[ , 1] = rep_vector(1,N);

  for (i in 1:N) {
    x_vec[i] = x[i];
    noise_matrix[i,2] = x[i];
    noise_matrix[i,3] = x[i]^2;
    noise_matrix[i,4] = x[i]^3;
  }

  noise_matrix_predict[ , 1] = rep_vector(1,N_predict);

  for (i in 1:N_predict) {
    x_vec_predict[i] = x_predict[i];
    noise_matrix_predict[i,2] = x_predict[i];
    noise_matrix_predict[i,3] = x_predict[i]^2;
    noise_matrix_predict[i,4] = x_predict[i]^3;
  }

}

parameters {
  real<lower=0,upper=0.25> rho_mu;
  real<lower=0> alpha_mu;

  real<lower=-0.5,upper=0.5> xi;

  matrix[4,1] beta_lin;

  vector[N] mu_latent;
}

transformed parameters {
  matrix[N,1] sigma;

  sigma = exp(noise_matrix * beta_lin);

}

model {
  matrix[N, N] cov_mu =   cov_exp_quad(x, alpha_mu, rho_mu) + diag_matrix(rep_vector(1e-06, N));
  matrix[N, N] L_cov_mu = cholesky_decompose(cov_mu);

  real t_x[N];

  rho_mu ~ normal(0, 20.0 / 3);
  alpha_mu ~ normal(0, 2);

  xi ~ normal(0,1);

  mu_latent ~ multi_normal_cholesky(rep_vector(0, N), L_cov_mu);

  for (i in 1:N) {
    if (i == 0 ) {
      t_x[i] = exp(-(y[i]-mu_latent[i])/sigma[i,1]);
      } else {
      t_x[i] = (1 + xi * (y[i] - mu_latent[i]) / sigma[i,1])^(-1/xi);
    }
    target += - log(sigma[i,1]) + (xi + 1) * log(t_x[i]) - (t_x[i]);
  }

}

generated quantities {
  vector[N_predict] mu_predict = gp_pred_rng(x_predict, mu_latent, x, alpha_mu, rho_mu, 1e-10);
  vector[N_predict] sigma_predict = to_vector(exp(noise_matrix_predict * beta_lin));

}
