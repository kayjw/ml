# Model

## Introduction

A real astete eagent might say that he estimates houses by intuition. But on closer inspection, you can see that he recognizes price patterns and uses them to predict new houses.  
Machine learning works the same way.

The Decision Tree is one of the many models.  
It may not be as accurate in predicting as others, but easy to understand and the building block for some of the best models in data science.

This example groups houses into two groups (with price predictions).

        -----------------------> $1.100.000
      /
     / 3 or more
    * bedrooms
     \ less than 3
      \
        -----------------------> $750.000

Training data is then used to fit the model. Which training/adjusting it so that the splits and endprices are as optimal as possible.  
After that, it can be used to predict the prices of new data.

The results of the above tree would be rather vague.  
By using deeper trees (with more splits) you can capture more factors.

    								  ----> $1.300.000
    								/
                                   / yes
          ----------------------- * larger than 11500 square feet lot size
        /                          \ no
       /                            \
      /                               ----> $750.000
     / 3 or more
    * bedrooms
     \ less than 3
      \                               ----> $850.000
       \                            /
        \						   / yes
    	  ----------------------- * larger than 8500 square feet lot size
    						       \ no
    							    \
    								  ----> $400.000

The point at the bottom, where the prediction is made, is called leaf.
