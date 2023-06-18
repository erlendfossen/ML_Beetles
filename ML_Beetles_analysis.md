Using machine learning to predict/classify species of beetles
================

## Data

Explanation of variables in the data:

- **ID**: ID of the beetle specimen.

- **morphotype**: The species of beetle. All within the genus Hydrobius.
  This is the outcome I want to predict.

- **15 morphological features**: Includes body size, length and width of
  the wings and pronotum, and smaller structures on the males. Most
  measured in mm or μm. Some features are ratios between two lengths.
  One feature characterizing the shape of a structure (mesoShape) is an
  angle that is measured in degrees. See publication for details.

Looking at the number of samples per species.

``` r
table(beetle_data$morphotype)
```


        arcticus     fuscipes rottenbergii  subrotundus 
              22           39           27           35 

Next, looking at how much missing data there is per variable:

``` r
# Missing per variables
naniar::vis_miss(beetle_data)
```

![](ML_Beetles_analysis_files/figure-gfm/missingness-1.png)<!-- -->

``` r
# Number of specimens with complete data
table(beetle_data[complete.cases(beetle_data), "morphotype"])
```

    morphotype
        arcticus     fuscipes rottenbergii  subrotundus 
               4            3            3            3 

As seen, there are very few complete cases, primarily do high
missingness in variables that were difficult/time-consuming to measure.
Not ideal, but can imputation to fill in likely missing values.
