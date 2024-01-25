#include <complex.h>
#include <math.h>
#include <stdio.h>

void evaluate(double complex *z1_true, double complex *z2_true, double *alpha, int N, int accurate) {
    FILE *fp;
    if (accurate){
        fp = fopen("Accurate.txt", "w");
    } 
    else{
        fp = fopen("Exact.txt", "w");
    }
    
    float complex z1, z2;

    for (int j = 0; j < N; j++) {
        float alp = (float)alpha[j];
        float complex y;
        
        if (fabsf(alp) < 1){
            y = I*sqrtf(1 - alp*alp);
        }
        
        else{
            y = sqrtf(alp*alp - 1);
        }

        if (accurate) {
            if (alp < 0) {
                z1 = -alp + y;
                z2 = 1 / z1;
            } else {
                z2 = -alp - y;
                z1 = 1 / z2;
            }
        } else {
            z1 = -alp + y;
            z2 = -alp - y;
        }

        fprintf(fp, "%.15f\t", alpha[j]);
        fprintf(fp, "%.15f\t", cabs(z1 - z1_true[j]));
        fprintf(fp, "%.15f\t", cabs(z2 - z2_true[j]));
        fprintf(fp, "%.15f\t", cabs(z1 - z1_true[j]) / cabs(z1_true[j]));
        fprintf(fp, "%.15f\n", cabs(z2 - z2_true[j]) / cabs(z2_true[j]));
    }

    fclose(fp);
}

int main() {
    double a_start = 1, a_end = 10000;
    int N = 10000;

    double alpha[N];
    for (int i = 0; i < N; i++) {
        alpha[i] = a_start * pow(a_end / a_start, (double)i / (N - 1));
    }

    double complex z1_true[N], z2_true[N];
    for (int i = 0; i < N; i++) {
        double complex y;

        if (fabs(alpha[i]) < 1){
            y = I*sqrt(1 - alpha[i]*alpha[i]);
        }

        else{
            sqrt(alpha[i] * alpha[i] - 1);
        }

        z1_true[i] = -alpha[i] + y;
        z2_true[i] = -alpha[i] - y;
    }

    evaluate(z1_true, z2_true, alpha, N, 0);
    evaluate(z1_true, z2_true, alpha, N, 1);

    return 0;
}