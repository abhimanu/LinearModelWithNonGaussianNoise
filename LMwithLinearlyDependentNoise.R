# Author Abhimanu Kumar
# This R script needs forecast package
# To run this script type source(".R")

# discard warnings to avoid clutter on console
#options(warn=-1)

# This calculates the negative log likelihood.
# This is also the error value that we 
# are minimizing in gradient descent
# b0,b1,var1, and var2 are beta_0, beta_1
# sigma_1^2 and sigma_2^2 respectively.
# The parameter simpleModel is a flag (0,1)
# that tells whether to fix \sigma^2_2 at 0
# or learn it.
neg_ll <- function(b1,b0,var1,var2, dataset, simpleModel){
	totalError = 0 
    for(i in 1:nrow(dataset)){
        x = dataset[i, 1]
        y = dataset[i, 2]
        noiseVar=var1*x**2+ (1-simpleModel)*var2
        totalError = totalError + .5*(y - (b1 * x + b0))**2/(noiseVar)+.5*log(noiseVar)    
	}
    return(totalError)	
}

# This function calculates the gradient update for each epoch of 
# gradient descent. It calculate the gradient updates for each data points
# and sums all these updates to obtain the final update.
# the learning_rate is the rate of learning for the gradient descent 
# During the updates if the value of variance goes below precision 
# we boost it back up to minimum machine precision
gradient_update <- function(b1,b0,var1,var2, dataset, learning_rate, simpleModel){
	b1_grad=0
    b0_grad=0
    var1_grad=0
    var2_grad=0
    for (i in 1:nrow(dataset)){
        xi = dataset[i, 1]
        yi = dataset[i, 2]
        noiseVar=var1*xi**2+ (1-simpleModel)*var2
        b1_grad = b1_grad -xi*(yi-b1*xi-b0)/(noiseVar)
        b0_grad = b0_grad -(yi-b1*xi-b0)/(noiseVar)
        var1_grad = var1_grad + .5*(xi**2/noiseVar)-.5*(xi**2)*(yi-b1*xi-b0)**2/(noiseVar)**2
        if (simpleModel<=0)
            var2_grad = var2_grad + .5*(1/noiseVar)-.5*(yi-b1*xi-b0)**2/(noiseVar)**2
	}

    b1 = b1 - (learning_rate * b1_grad)
    b0 = b0 - (learning_rate * b0_grad)
    var1 = var1 - (learning_rate * var1_grad)
    var2 = var2 - (learning_rate * var2_grad)

    var1=max(var1,.Machine$double.xmin)
    var2=max(var2, .Machine$double.xmin)

    return(c(b1, b0, var1,var2))
}

# This function calculates the residue or the noise
# given the dataset and final model weights. It assumes
# that \sigma^2_2 is 0 as is the case with dataset1 and 
# dataset2
residue <- function(b1,b0,var1,var2, dataset, simpleModel){
	return((dataset[,2]-(b1*dataset[,1] + b0))/dataset[,1])
}

# This function performs the EDA or the 
# Exploratory Data Analysis.
# It reports the point plots and linear model
# fit. It also reports the normality tests for the 
# residue of the linear model.
# Finally it also reports correlation 
# between x and y with and without "outliers"  
eda <- function(filename){
    data1 = read.csv(filename, header=T)
	summary(data1)
	fit1 = lm(y~x, data=data1)
	setEPS()
	print(paste("correlation between x and y =",cor(data1$x,data1$y)))
	# the below is only valid for dataset1
	print(paste("correlation between x and y after removing 'outliers' =",
		cor(data1[data1$y<20 & data1$y>-20,]$x,data1[data1$y<20 & data1$y>-20,]$y)))
	filenameShort = strsplit(filename, '/')[length(strsplit(filename, '/'))]
	postscript(paste("fitPlot_",filenameShort,".eps",sep=''))
	plot(data1$x, data1$y, main='dataset1 data points and plot of linear fit y~x', xlab='X', ylab='Y')
	lines(data1$x, data1$x*fit1$coefficients[2] + fit1$coefficients[1], col='red')
	dev.off()
	print(paste("Jarque Bera Normality test for Gaussian noise model for", filenameShort))
	residue_LM = rstandard(fit1)
	print(jarqueberaTest(residue_LM))
	print(paste("Shapiro - Wilk Normality test for Gaussian noise model for", filenameShort))
	print(shapiroTest(residue_LM))
	print(paste("D'Agostino Normality test for Gaussian noise model for", filenameShort))
	print(dagoTest(residue_LM))
}

# This is the main function that performs the gradient 
# descent to obtain b1 (\beta_1), b0 (\beta_0), var1 (\sigma^2_1), 
# and var2 (\sigma^2_2) for the new model with linearly dependent noise
# assumption. It also reports the normality tests for the 
# residues of the new model. The argument simpleModel (flag with value 
# 1 or 0) decides whetehr to learn \sigma^2_2 or fix it to 0. 
lmNonGaussian <- function(filename, simpleModel=0){
    #  load data from csv file
	
    dataset = read.csv(filename, header=T)
    X=dataset[,1]
    Y=dataset[,2]
	#dataSorted = dataset[with(dataset, order(x)),]
	threshold = 10**(-5)
	thresh_count = 10

    # plot y vs x
	setEPS()
	filenameShort = strsplit(filename, '/')[length(strsplit(filename, '/'))]
	postscript(paste("fitPlot_",filenameShort,".eps",sep=''))
    plot(X,Y, xlab='X', ylab='Y')


    learning_rate = 0.001 # learning rate
    num_iterations=3000

    # initialize parameters
    b1 = runif(1,0,1)
    b0 = runif(1,0,1)
    var1 = runif(1,0,1)
    var2 = runif(1,0,1)*(1-simpleModel)
    initial_neg_ll= neg_ll(b1,b0,var1,var2, dataset, simpleModel);
	print (paste("nrow(dataset)",nrow(dataset)))

    print(sprintf("Initial values: b1 = %f, b0 = %f, var1 = %f, var2 = %f, neg_ll = %f",b1, b0, var1,var2, initial_neg_ll))

    # gradient descent running
	prev_b1 = b1; prev_b0 = b0;
	thresh_b1_count = 0; thresh_b0_count = 0;
	for(i in 1:num_iterations){
        c_return = gradient_update(b1, b0,var1,var2, dataset, learning_rate, simpleModel)
        b1 = c_return[1]; b0 = c_return[2]; var1 = c_return[3]; var2 = c_return[4];
        final_neg_ll= neg_ll(b1,b0,var1,var2, dataset,simpleModel);
    	print(sprintf("Iterations = %d, b1 = %f, b0 = %f, var1 = %f, var2 = %f, neg_ll = %f", i, b1, b0, var1,var2, final_neg_ll))
		if(abs((prev_b1-b1)/b1) < threshold) {
			thresh_b1_count = thresh_b1_count+1
		}
		if(abs((prev_b0-b0)/b0) < threshold) {
			thresh_b0_count = thresh_b0_count+1
		}
		if(thresh_b0_count > thresh_count && thresh_b1_count > thresh_count) {
			break
		}
		prev_b1 = b1; prev_b0 = b0
	}

	# plot the fitted curve
	lines(X,b1*X+b0,col='blue')

	# fit a simple linear model with gaussian noise
	fitLM = lm(y~x, data=dataset)
	# plot the fitted LM
	lines(X, X*fitLM$coefficients[2] + fitLM$coefficients[1], col='red')
	legend(min(X)+(max(X)-min(X))/4.0,min(Y)+(max(Y)-min(Y))/4.0, c("Gaussian Noise","Non-Gaussian Noise"), 
		lty=c(1,1), lwd=c(2.5,2.5),col=c("red","blue"))
	dev.off()

    # variance of estimated b1 and b0
    temp_b1=0
    temp_b0=0
    for (i in 1:nrow(dataset)){
        xi = dataset[i, 1]
        noiseVar=var1*xi**2+ (1-simpleModel)*var2
        temp_b1 = temp_b1 + xi**2.0/noiseVar
        temp_b0 = temp_b0 + 1.0/noiseVar
	}
    var_b1=1/temp_b1
    var_b0=1/temp_b0
    print(sprintf("variance of b1 %f and b0 %f ",var_b1, var_b0))
    require('fBasics')
	residue_LD = residue(b1,b0,var1,var2, dataset, simpleModel)
	#cat(residue_LD,sep='\n')
	#write(residue_LD, file='temp_residue.csv')
	#unlink('temp_residue.csv')
	print(paste("Jarque Bera Normality test for linearly dependent noise model for", filenameShort))
	print(jarqueberaTest(residue_LD))
	print(paste("Shapiro - Wilk Normality test for linearly dependent noise model for", filenameShort))
	print(shapiroTest(residue_LD))
	print(paste("D'Agostino Normality test for linearly dependent noise model for", filenameShort))
	print(dagoTest(residue_LD))
}


