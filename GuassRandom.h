double _random()
{
	int a;
	double r;
	a=rand()%32767;
	r=(a+0.00)/32767.00;
	return r;
}
double GetOneGaussian(double mu,double sigma)
{
	double r1,r2;
	r1 = _random();
	r2 = _random();
	return sqrt(-2*log(r1))*cos(2*3.1415*r2)*sigma+mu;
}