#include <stdio.h>

//extern "C"
//{
//  #include <cblas.h>
//}
//#include <cblas.h>
#include <gsl/gsl_cblas.h>

double m[] = {
  3, 1, 3,
  1, 5, 9,
  2, 6, 5
};

double matrix1[] = {
  1, 1, 1,
  1, 1, 1,
  1, 1, 1
};

double matrix2[] = {
  2, 2, 2,
  2, 2, 2,
  2, 2, 2
};

double matrix3[] = {
  2, 2, 2,
  2, 2, 2,
  2, 2, 2
};

double x[] = {
  -1, -1, 1
};

double y[] = {
  0, 0, 0
};

int main() {
	int i, j;

	printf("===> matrix1\n");
	for (i=0; i<3; ++i) {
		for (j=0; j<3; ++j) printf("%5.1f", matrix1[i*3+j]);
			putchar('\n');
	}
	printf("===> matrix2\n");
	for (i=0; i<3; ++i) {
		for (j=0; j<3; ++j) printf("%5.1f", matrix2[i*3+j]);
			putchar('\n');
	}

//DGEMV ( TRANS, M, N, ALPHA, A, LDA, X, INCX,BETA, Y, INCY )
//  cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, m, 3,x, 1, 0.0, y, 1);

	printf("===> x\n");
	for (i=0; i<3; ++i)  printf("%5.1f\n", x[i]);
	printf("===> y\n");
	for (i=0; i<3; ++i)  printf("%5.1f\n", y[i]);

	printf("===> matrix3\n");
	cblas_dgemm (CblasRowMajor,CblasNoTrans,CblasNoTrans,3,3,3,1.0,matrix1,3,matrix2,3,0.0,matrix3,3);
//DGEMM ( TRANSA, TRANSB, M, N, K, ALPHA, A      , LDA,B      , LDB, BETA, C      , LDC )
//DGEMM ( 'N'     , 'N'    , 3, 3, 3, 1.0  , matrix1, 1  ,matrix2, 1  , 0.0 , matrix3, 1   );
//cblas_dgemm ( 'N'     , 'N'    , 3, 3, 3, 1.0  , matrix1, 3  ,matrix2, 3  , 0.0 , matrix3, 3   );

	for (i=0; i<3; ++i) {
		for (j=0; j<3; ++j) printf("%5.1f", matrix3[i*3+j]);
		putchar('\n');
	}

	return 0;
}

