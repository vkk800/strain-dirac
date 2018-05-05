#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define M_PI           3.14159265358979323846

typedef struct _fcomplex { double re, im; } fcomplex;

extern void zheevd_(char* jobz, char* uplo, int* n, fcomplex* a, int* lda,
		    double* w, fcomplex* work, int* lwork, double* rwork, int* lrwork,
		    int* iwork, int* liwork, int* info);

void initmatrix(fcomplex* M, int* nm) {
	int r,c;
	fcomplex num = {0.0, 0.0};
	for (r=0 ; r < (2*(*nm)+1)*4 ; r++) {
		for (c=0 ; c < (2*(*nm)+1)*4 ; c++) {
			M[r*(2*(*nm)+1)*4+c] = num;
		}
	}
}

void constructVDmat(int* restrict nm, fcomplex* restrict vn3, fcomplex* restrict dn,
		    fcomplex* restrict M) {
	int r,c;
	int i,j;
	int nsize = 2*(*nm)+1;
	int multr = nsize*4*4;
	int multc = 4;
	int multi = nsize*4;
	for (r=0 ; r < (2*(*nm)+1) ; r++) {
		for (c=0 ; c < (2*(*nm)+1) ; c++) {
			if (abs(r-c)<((*nm)+1)) {
				// The potential term
				M[multr*r+multc*c+1].re += vn3[r-c+(*nm)].re;
				M[multr*r+multc*c+1].im += vn3[r-c+(*nm)].im;
				M[multr*r+multc*c+multi].re += vn3[r-c+(*nm)].re;
				M[multr*r+multc*c+multi].im -= vn3[r-c+(*nm)].im;
				M[multr*r+multc*c+multi*2+3].re -= vn3[r-c+(*nm)].re;
				M[multr*r+multc*c+multi*2+3].im -= vn3[r-c+(*nm)].im;
				M[multr*r+multc*c+multi*3+2].re -= vn3[r-c+(*nm)].re;
				M[multr*r+multc*c+multi*3+2].im += vn3[r-c+(*nm)].im;

				// The delta term
				M[multr*r+multc*c+2].re += dn[r-c+(*nm)].re;
				M[multr*r+multc*c+2].im += dn[r-c+(*nm)].im;
				M[multr*r+multc*c+multi+3].re += dn[r-c+(*nm)].re;
				M[multr*r+multc*c+multi+3].im += dn[r-c+(*nm)].im;
				M[multr*r+multc*c+multi*2].re += dn[c-r+(*nm)].re;
				M[multr*r+multc*c+multi*2].im -= dn[c-r+(*nm)].im;
				M[multr*r+multc*c+multi*3+1].re += dn[c-r+(*nm)].re;
				M[multr*r+multc*c+multi*3+1].im -= dn[c-r+(*nm)].im;
			}
		}
	}
}

void replaceBdG(int* restrict nm, fcomplex* restrict M, double* restrict kx, double* restrict ky) {
	int r,c;
	int i,j;
	int nsize = 2*(*nm)+1;
	int multr = nsize*4*4;
	int multc = 4;
	int multi = nsize*4;

	for (r=0 ; r < (2*(*nm)+1) ; r++) {
		fcomplex num = {(*kx)+2*M_PI*(r-(*nm)), -(*ky)};
		M[multr*r+multc*c+1] = num;
		fcomplex num2 = {(*kx)+2*M_PI*(r-(*nm)), (*ky)};
		M[multr*r+multc*c+multi] = num2;
		fcomplex num3 = {-(*kx)-2*M_PI*(r-(*nm)), (*ky)};
		M[multr*r+multc*c+multi*2+3] = num3;
		fcomplex num4 = {-(*kx)-2*M_PI*(r-(*nm)), -(*ky)};
		M[multr*r+multc*c+multi*3+2] = num4;
	}
}

void constructBdG(int* restrict nm, fcomplex* restrict vn3, fcomplex* restrict dn, 
		  fcomplex* restrict M, double* restrict kx, double* restrict ky) {
	int r,c;
	int i,j;
	int nsize = 2*(*nm)+1;
	int multr = nsize*4*4;
	int multc = 4;
	int multi = nsize*4;
	for (r=0 ; r < (2*(*nm)+1) ; r++) {
		for (c=0 ; c < (2*(*nm)+1) ; c++) {
			if (r==c) {
				fcomplex num = {(*kx)+2*M_PI*(r-(*nm)), -(*ky)};
				M[multr*r+multc*c+1] = num;
				fcomplex num2 = {(*kx)+2*M_PI*(r-(*nm)), (*ky)};
				M[multr*r+multc*c+multi] = num2;
				fcomplex num3 = {-(*kx)-2*M_PI*(r-(*nm)), (*ky)};
				M[multr*r+multc*c+multi*2+3] = num3;
				fcomplex num4 = {-(*kx)-2*M_PI*(r-(*nm)), -(*ky)};
				M[multr*r+multc*c+multi*3+2] = num4;
			}
			if (abs(r-c)<((*nm)+1)) {
				// The potential term
				M[multr*r+multc*c+1].re += vn3[r-c+(*nm)].re;
				M[multr*r+multc*c+1].im += vn3[r-c+(*nm)].im;
				M[multr*r+multc*c+multi].re += vn3[r-c+(*nm)].re;
				M[multr*r+multc*c+multi].im -= vn3[r-c+(*nm)].im;
				M[multr*r+multc*c+multi*2+3].re -= vn3[r-c+(*nm)].re;
				M[multr*r+multc*c+multi*2+3].im -= vn3[r-c+(*nm)].im;
				M[multr*r+multc*c+multi*3+2].re -= vn3[r-c+(*nm)].re;
				M[multr*r+multc*c+multi*3+2].im += vn3[r-c+(*nm)].im;

				// The delta term
				M[multr*r+multc*c+2].re += dn[r-c+(*nm)].re;
				M[multr*r+multc*c+2].im += dn[r-c+(*nm)].im;
				M[multr*r+multc*c+multi+3].re += dn[r-c+(*nm)+2*(*nm)+1].re;
				M[multr*r+multc*c+multi+3].im += dn[r-c+(*nm)+2*(*nm)+1].im;
				M[multr*r+multc*c+multi*2].re += dn[c-r+(*nm)].re;
				M[multr*r+multc*c+multi*2].im -= dn[c-r+(*nm)].im;
				M[multr*r+multc*c+multi*3+1].re += dn[c-r+(*nm)+2*(*nm)+1].re;
				M[multr*r+multc*c+multi*3+1].im -= dn[c-r+(*nm)+2*(*nm)+1].im;
			}
		}
	}
}

void solvedirac(int* restrict nm, fcomplex* restrict vn3, fcomplex* restrict dn, 
		fcomplex* restrict M, double* restrict kx, double* restrict ky,
		double* restrict eigE) {
	constructBdG(nm, vn3, dn, M, kx, ky);
	int n = (2*(*nm)+1)*4, info, lwork, lrwork, liwork;
	int iwkopt;
	int* iwork;
	double rwkopt;
	double* rwork;
        fcomplex wkopt;
        fcomplex* work;
        fcomplex z[n*n];

        lwork = -1;
        zheevd_( "Vectors", "Lower", &n, M, &n, eigE, &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info );
        lwork = (int)wkopt.re;
        work = (fcomplex*)malloc( lwork*sizeof(fcomplex) );
	lrwork = (int)rwkopt;
	rwork = (double*)malloc(lrwork*sizeof(double));
	liwork = iwkopt;
	iwork = (int*)malloc(liwork*sizeof(int));
        zheevd_( "Vectors", "Lower", &n, M, &n, eigE, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
}

void eigensolver(int* restrict nm, fcomplex* restrict M, double* restrict eigE) {
	int n = (2*(*nm)+1)*4, info, lwork, lrwork, liwork;
	int iwkopt;
	int* iwork;
	double rwkopt;
	double* rwork;
        fcomplex wkopt;
        fcomplex* work;
        fcomplex z[n*n];

        lwork = -1;
        zheevd_( "Vectors", "Lower", &n, M, &n, eigE, &wkopt, &lwork, &rwkopt, &lrwork, &iwkopt, &liwork, &info );
        lwork = (int)wkopt.re;
        work = (fcomplex*)malloc( lwork*sizeof(fcomplex) );
	lrwork = (int)rwkopt;
	rwork = (double*)malloc(lrwork*sizeof(double));
	liwork = iwkopt;
	iwork = (int*)malloc(liwork*sizeof(int));
        zheevd_( "Vectors", "Lower", &n, M, &n, eigE, work, &lwork, rwork, &lrwork, iwork, &liwork, &info);
}

void eigensolver_replace(int* restrict nm, fcomplex* restrict VDmat,
			 double* restrict eigE, double* restrict kx, double* restrict ky) {
	replaceBdG(nm, VDmat, kx, ky);
	eigensolver(nm, VDmat, eigE);
}

int dblcomp(const void* x, const void* y) {
	return (*(double*)x - *(double*)y);
}

void selfcintegrand(int* restrict nm, int* restrict nbands, double* restrict T,
		    fcomplex* restrict vn3, fcomplex* restrict dn, 
		    fcomplex* restrict M, double* restrict kx, double* restrict ky,
		    double* restrict eigE, fcomplex* newd) {
	solvedirac(nm, vn3, dn, M, kx, ky, eigE);
	int b, n, m;
	int multr = (2*(*nm)+1)*4;
	for (b=4*(*nm)-2*(1-(*nbands)) ; b < 4*(*nm)+1+2*(1+(*nbands)) ; b++) {
		for (n=0 ; n < 2*(*nm)+1 ; n++) {
			for (m=0 ; m < 2*(*nm)+1 ; m++) {
				if (!((m-n)>=-(*nm))) {
					continue;
				}
				if (!((m-n)<((*nm)+1))) {
					continue;
				}
				newd[n].re += (M[b*multr+((m-n+(*nm))*4+2)].re*M[b*multr+(m*4)].re
					       +M[b*multr+((m-n+(*nm))*4+2)].im*M[b*multr+(m*4)].im)
					*tanh(eigE[b]/(2*(*T)));
				newd[n].im -= (M[b*multr+((m-n+(*nm))*4+2)].re*M[b*multr+(m*4)].im
					       -M[b*multr+((m-n+(*nm))*4+2)].im*M[b*multr+(m*4)].re)
					*tanh(eigE[b]/(2*(*T)));
				
				newd[n+2*(*nm)+1].re += (M[b*multr+((m-n+(*nm))*4+3)].re*M[b*multr+(m*4+1)].re
							 +M[b*multr+((m-n+(*nm))*4+3)].im*M[b*multr+(m*4+1)].im)
					*tanh(eigE[b]/(2*(*T)));
				newd[n+2*(*nm)+1].im -= (M[b*multr+((m-n+(*nm))*4+3)].re*M[b*multr+(m*4+1)].im
							 -M[b*multr+((m-n+(*nm))*4+3)].im*M[b*multr+(m*4+1)].re)
					*tanh(eigE[b]/(2*(*T)));
			}
		}
	}	
}
