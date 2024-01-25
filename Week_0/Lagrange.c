#include <stdio.h>
#include <math.h>
#include <stdlib.h>

float lintp(float *xx, float *yy, float x, int n){
    double ret = 0;

    for(int i=0;i<=n;i++){
        double L = 1;
        for(int j=0;j<=n;j++) if(i!=j) L*= (x-xx[j])/(xx[i]-xx[j]);

        ret += L * yy[i];
    }

    return ret;
}


int main(int argc, char **argv){
    char *name = argv[1];
    
    int flag = atoi(argv[2]);

    float x = 0, y, pi = 3.1415926;

    float xx[] = {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2};
    float yy[9], yyNoise[13][9];

    for(int i=0;i<9;i++) yy[i] = sin(pi*xx[i]);

    FILE *noiseFile, *output;
    output = fopen("output.txt", "w");
    noiseFile = fopen(name, "r");
    
    for(int i=0;i<13;i++)
        for(int j=0;j<9;j++)
            fscanf(noiseFile, "%f,", &yyNoise[i][j]);

    for(int i=0;i<=100;i++){
        if(flag == 0) y = lintp(xx, yy, x, 8);

        else y = lintp(xx, yyNoise[flag-1], x, 8);

        fprintf(output, "%f %f\n", x, y);

        x += 0.02; // 100 uniform samples in (0, 2)
    }
    
    return 0;
}


