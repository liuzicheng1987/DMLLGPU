class LogLossCpp: public LossFunctionCpp {
	
	public:	
	
	//Constructor
	LogLossCpp (): LossFunctionCpp() {
		
	}

	//Virtual destructor		
	~LogLossCpp() {}
		
void Loss(MPI_Comm comm, const int rank, const int size, const int I, const int J, double *loss, const double *Y, const double *Yhat) {
			

}
	
void dLossdYhat (MPI_Comm comm, const int rank, const int size, const int I, const int J, const int skipJ, double *dlossdYhat, const double *Y, const double *Yhat) {
	
	int i, j;
	
	for (i=0; i<I; ++i) for (j=0; j<J; ++j) dlossdYhat[i*(J + skipJ) + j] = (1.0 - Y[i*J + j])/(1.0 - Yhat[i*J + j]) - Y[i*J + j]/Yhat[i*J + j];
}
	
};
