# Forecasting-Product-Demand

## Project/Goals

The problem we are considering is that of forecasting product demand for a collection of Walmart stores. We are thus faced with a time-series problem. 
2.	Dataset
The dataset contains the unit sales of various products sold in the USA organizes in a grouped time series. More specifically, the datset contains the unit sales of 30,49 products, classified into three product categories (Hobbies, Foods, and Households) and are in turn disaggregated into 7 product departments per store. The products are sold across 10 stores located in 3 state (CA, TX, WI). 
The bottom level of the grouped data structure is a product – dept – category – store – state combination. Importantly this data structure is not hierarchical because there are multiple aggregation paths, across varying combinations of product and geographical mappings. 
[Aggregation Paths] Unit Sales of all products can be aggregated and disaggregated by:
•	Total (all stores/states)
•	State
•	Store
•	Category
•	Department
•	Category – State Combination
•	Department – State Combination
•	Category-Store
•	Department Store
Unit sales of some product x can be also aggregated/disaggregated by:
•	Store – State Combination
•	Store
•	State
In other words, we can aggregate/disaggregate along either:
(1)	Geographic lines
(2)	Product Lines
(3)	Crossing of product and geographic lines. 
There are also companion datasets with some explanatory variables like: sale price, promotions, days of the week, and special events
