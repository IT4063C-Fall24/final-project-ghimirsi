# Exploring Factors Influencing Online Movie Ratings
<!-- Edit the title above with your project title -->

## Project Overview
This project explores key factors influencing online movie ratings, focusing on genre, release year, and production budget. Using data from IMDb, Netflix, and TMDB, the analysis aims to uncover patterns and correlations that reveal how these elements affect audience reception. By examining average ratings across genres, identifying trends over time, and analyzing the relationship between budget and ratings, the project seeks to provide valuable insights for movie producers and streaming platforms. The findings will be presented through visualizations and summarized to offer actionable recommendations on content creation and marketing strategies.

## Self Assessment and Reflection

<!-- Edit the following section with your self assessment and reflection -->

### Self Assessment
<!-- Replace the (...) with your score -->

| Category          | Score    |
| ----------------- | -------- |
| **Setup**         | .10. / 10 |
| **Execution**     | .20. / 20 |
| **Documentation** | .10. / 10 |
| **Presentation**  | .30. / 30 |
| **Total**         | .70. / 70 |

### Reflection
<!-- Edit the following section with your reflection -->

#### What went well?
The data preparation and merging process was quite seamless, which allowed for the successful integration of the various datasets I worked with. The exploratory data analysis (EDA) provided valuable insights into consumption patterns and helped pinpoint key demographic influences. I found that the visualizations I created effectively conveyed the findings, making the data more accessible and easier to understand for others.

#### What did not go well?
I did face some initial hurdles with missing values and discrepancies in demographic identifiers, which took extra time to clean and standardize the datasets properly. Additionally, certain statistical analyses turned out to be more complex than I had anticipated, which delayed my ability to interpret the results effectively.

#### What did you learn?
This project underscored the critical importance of thorough data cleaning and preparation prior to analysis, as it greatly influences the quality of the insights I can derive. I also gained hands-on experience in merging datasets from different sources and conducting regression analysis to uncover relationships between variables, which was a valuable learning experience.

#### What would you do differently next time?
In future projects, I would allocate more time specifically for the data cleaning phase and develop a more detailed plan for addressing missing values and discrepancies in the datasets. I would also consider exploring additional data sources earlier in the process to enhance my analysis. Finally, I would document my analysis process more rigorously to facilitate smoother workflows in future projects and improve reproducibility.
---

## Getting Started
### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bash
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Helpful Resources:
* [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
* [Dataset options](https://it4063c.github.io/guides/datasets)