#import "@preview/typslides:1.2.5": *

// Project configuration
#show: typslides.with(
  ratio: "16-9",
  theme: "dusky",
)

// The front slide is the first slide of your presentation
#front-slide(
  title: "Love thy neighbor?",
  subtitle: [#set text(size: 22pt)
    An empirical test of neighborhood ethnicity change and Schelling behavior],
  authors: "Jørgen Baun Høst",
  info: [University of Copenhagen · Department of Economics · May 2025],
)

// Custom outline
#slide(title: "Overview")[
  #set text(size: 22pt)
  - *Motivation*: Residential segregation in Denmark and Schelling's model
  - *Research Question*: Does the ethnicity of your nearest neighbor affect propensity to move?
  - *Methods*: Nearest-neighbor research design with comprehensive administrative data
  - *Results*: Asymmetry in residential responses based on ethnicity
  - *Heterogeneity Analysis*: SES differences in Schelling behavior
  - *Conclusion*: Evidence of individually motivated segregation
]

// Introduction and motivation
#title-slide[
  Introduction
]

#slide(title: "To do")[
  #set text(size: 22pt)
  - What should go in appendix and what should be in prez?
  - Exclude data sources(?)
  - Spend time on spatial patterns. This
]

#slide(title: "Residential Segregation in Denmark")[
  #set text(size: 22pt)
  #cols(columns: (1fr, 1fr), float: true)[
    - Denmark has transformed from a relatively homogeneous society to increasing ethnic diversity
    - Non-Western households grew from ~2% in 1985 to ~10% by 2020
    - Limited empirical evidence on how ethnic background directly influences residential sorting
  ][
    // #image("https://source.unsplash.com/random/800x500/?denmark+housing", width: 100%)
    #text(size: 10pt)[Placeholder image - replace with actual data visualization of segregation patterns]
  ]
]

#slide(title: "Theoretical Background: Schelling's Model")[
  #set text(size: 22pt)
  - #strong[@schelling1971dynamic] proposed that neighborhoods may "tip" when minority share reaches a threshold
  - Even with relatively tolerant preferences toward diversity
  - Three types of segregation:
    1. Organized segregation (e.g., historical Jim Crow laws)
    2. Economically induced segregation (clustering by income/education)
    3. #reddy[Individually motivated segregation] ← #strong[Focus of this paper]

  - Schelling's key insight: Small individual preferences can lead to macro-level segregation
]

// Methods
#title-slide[
  Methods
]

#slide(title: "Empirical Framework")[
  #set text(size: 22pt)
  Modeling a household's decision to stay or move in a neighborhood that evolves over time:

  $ U_(i,j,t) = f(Z_(i,t), X_(j,t), xi_(j,t)) + sum_(k) g(Z_i, Z_(k,t), D_(i,k)) + epsilon_(i,j,t) $

  Where:
  - $f(·)$: Utility from neighborhood amenities
  - $g(·)$: Utility from characteristics of each neighbor $k$ at distance $D_(i,k)$
  - $Z_i$: Observable household attributes
  - $X_j$: Observable neighborhood attributes
  - $xi_j$: Unobservable neighborhood attributes
  - $epsilon_(i,j,t)$: Idiosyncratic preferences
]

#slide(title: "Identification Challenge")[
  #set text(size: 22pt)
  $
    V_(i,j,t) = f(Z_(i,t), X_(j,t), xi_(j,t), alpha) + sum_(k) g(Z_(i,t), Z_(k,t), D_(i,k), beta) + delta E[V_(i,j,t+1)] + epsilon_(i,j,t)
  $


  #strong[Key identification challenges:]
  - Unobserved neighborhood amenities
  - Dynamic preferences (expectations of future changes)
  - Selection effects (who moves where is not random)
]

#slide(title: "Nearest-Neighbor Research Design")[
  #set text(size: 18pt)
  #strong[Innovative approach from @Bayer_2022_nearest_neighbor:]

  Compare households within the same neighborhood who receive different-type neighbors:
  - #strong[Treatment group]: Households with new different-type neighbors among their 3 nearest neighbors
  - #strong[Control group]: Households with new different-type neighbors "just down the road" (ranks 4-6)

  #v(1em)
  #align(center)[

    $
      Y_(i,j,t) = beta_1 I[e', k = n_(n e a r e s t)] + beta_2 I[e', k = n_(n e a r)] + beta_3 I[e', k = n_(c l o s e)] + gamma Z_(i,j,t) + omega_(j,t) + epsilon_(i,j,t)
    $
    #reddy[Parameter of interest:] $ beta_1 - beta_2 $
  ]
  This design addresses key identification challenges by comparing households experiencing same neighborhood conditions but different micro-geography of new neighbors.
]

#slide(title: "Data Sources")[
  #set text(size: 22pt)
  #strong[Comprehensive Danish administrative data, 1985-2020:]

  - Population register (BEF): Demographics, family structure, country of origin
  - Income register (IND): Gross income, net wealth
  - Labor market register (RAS): Employment status
  - Education register (UDDF): Educational attainment

  #v(1em)
  #strong[Unique geospatial data (BOPAEL_KOORD):]
  - Precise coordinates with start/end dates at each address
  - 4-dimensional: $(x_E, y_N, z_F, z_D)$ (East, North, Floor, Door)
  - Enables construction of exact nearest neighbors for each household
]

// #slide(title: "Household Definitions and Sample Restrictions")[
//   #set text(size: 18pt)
//   #cols(columns: (1fr, 1fr))[
//     #strong[Household definition:]
//     - Graph theory approach using spatio-temporal overlap
//     - Maximizing edge weights to identify "stable" households
//     - ~14 million historical household sequences

//     #strong[Sample restrictions:]
//     - Income: 200,000-1,000,000 DKK (equivalized)
//     - Wealth: -200,000 to 750,000 DKK (equivalized)
//     - Age: Oldest member between 30-60
//     - Focus on native and non-Western households

//     #strong[Proximity requirements:]
//     - Nearest neighbor within 25 meters
//     - Neighborhoods with 1,000-25,000 people per km²
//     - 3,451 unique neighborhoods

//     #strong[Household types:]
//     1. Native: All members of Danish origin
//     2. Non-Western: At least one member of non-Western origin
//     3. Western: At least one non-native of Western origin, no non-Western members
//   ]]

// Results
#title-slide[
  Results
]

#slide(title: "Spatial Patterns of New Different-Type Neighbors")[
  #set text(size: 18pt)
  #cols(columns: (1fr, 1fr))[
    #strong[Key spatial patterns:]

    - Clear east-west and urban-rural divide
    - Concentration in Copenhagen and surroundings
    - Highest incidence in Ishøj (~9 new different-type neighbors)
    - Copenhagen (~6), Aarhus and Odense (~4)

    #strong[Within-city variation:]
    - Some Copenhagen neighborhoods: 30+ new non-Western neighbors
    - Other Copenhagen neighborhoods: < 2 new non-Western neighbors
  ][
    //#image("https://source.unsplash.com/random/800x500/?map+denmark", width: 100%)
    #text(size: 10pt)[Placeholder - replace with actual visualization from Figure 3]

    #v(1em)
    //#image("https://source.unsplash.com/random/800x500/?urban+map", width: 100%)
    #text(size: 10pt)[Placeholder - replace with actual visualization from Figure 4]
  ]
]

#slide(title: "Summary Statistics")[
  #set text(size: 18pt)
  #strong[Key observations from summary statistics:]

  - "Treated" households show higher mobility: 23-24% vs. 19-20% for "control" households
  - Treated native households have lower wealth (48,500 DKK vs. 81,000 DKK) and income
  - Treated non-Western households have slightly lower wealth than controls
  - Non-Western households are better educated on average (by ~2 years)
  - Native households tend to live in less dense, more affluent, less integrated neighborhoods
  - Treated native households live in neighborhoods with ~15% non-Western share vs. 8% for all native households

  #v(0.5em)
  #align(center)[
    These patterns highlight selection effects and the importance of the nearest-neighbor research design.
  ]
]

#slide(title: "Main Results: Asymmetric Schelling Behavior")[
  #set text(size: 18pt)
  #cols(columns: (1fr, 1fr), align: center)[
    = Native households
  ][
    = Non-Western households
  ]
  #cols(columns: (0.5fr, 0.5fr), gutter: 2em)[
    - Increase moving propensity by ~0.3 percentage points when receiving a new non-Western neighbor
    - 1.6% increase relative to baseline exit rate
    - Effect stable across specifications
    - Robust to controls for income, wealth, age, tenure
  ][
    - Show substantially smaller response: 0.06-0.1 percentage points
    - ~0.5% relative to baseline exit rate
    - Not statistically significant
    - Suggests they are unaffected by identity of new native neighbors
  ]

  #v(0.5em)
  #align(center)[
    #reddy[Key finding:] Asymmetric Schelling behavior in the Danish context
  ]
]

#slide(title: "Heterogeneity by Socioeconomic Status")[
  #set text(size: 18pt)
  #strong[SES definitions:]

  - #strong[Low SES]: Income < 200,000 DKK, outside labor market or $<=$ 11 years of education
  - #strong[High SES]: Income $>=$600,000 DKK, employed full-time or $>=$ 18 years of education

  #v(0.5em)
  #strong[Key findings:]

  - Schelling behavior primarily driven by low-SES native households responding to low-SES non-Western households
  - Effect size: ~0.56 percentage points or ~2.8% increase from baseline exit rate
  - Nearly twice the magnitude observed in full sample
  - Very rare for low-SES native households to receive high-SES non-Western neighbors and vice versa
  - Confirms powerful residential sorting at neighborhood level
]

#slide(title: "Comparison with U.S. Context")[
  #set text(size: 22pt)
  #strong[Danish findings vs. @Bayer_2022_nearest_neighbor U.S. results:]

  #align(center)[
    #table(
      columns: (auto, auto, auto),
      inset: 10pt,
      align: center,
      [*Context*], [*Response*], [*Magnitude*],
      [Denmark (Native)], [Asymmetric], [1.6% above baseline],
      [Denmark (Non-Western)], [Insignificant], [0.5% above baseline],
      [U.S. (White)], [Symmetric], [4% above baseline],
      [U.S. (Black)], [Symmetric], [6% above baseline],
    )
  ]

  #v(1em)
  #strong[Possible explanations for differences:]
  - Institutional variation in housing market and integration policies
  - Different neighborhood contexts (urban/dense vs. suburban)
  - Historical path dependence in residential patterns
]

// Conclusion
#title-slide[
  Conclusion
]

#slide(title: "Key Findings")[
  #set text(size: 22pt)
  1. Native Danish households increase moving propensity by 1.6% when receiving non-Western neighbors

  2. Non-Western households show no significant response to new native neighbors

  3. Heterogeneity by SES: Low-SES native households responding to low-SES non-Western neighbors show strongest effects (2.8%)

  4. Spatial decay of effects: Moving response decreases monotonically with distance to new different-type neighbors

  5. Magnitude in Denmark (1.6%) more modest than in U.S. context (4-6%)
]

#slide(title: "Implications and Contributions")[
  #set text(size: 22pt)
  #strong[Contributions to segregation research:]

  - Causal evidence of individually motivated segregation as theorized by Schelling (1971)
  - Demonstration of asymmetric responses in the European welfare state context
  - Socioeconomic gradient in responses highlighting intersection of ethnicity and economic resources
  - Evidence that Schelling mechanisms operate across different settings, but with context-specific magnitude and symmetry

  #v(1em)
  #strong[Policy implications:]
  - Integration efforts may need to account for micro-geography of neighborhood mixing
  - Targeted interventions may be more effective for low-SES populations
  - Understanding asymmetric responses could inform more effective housing policies
]

#focus-slide[
  Thank you for your attention!

  #v(1em)
  Questions?
]

// Bibliography
#let bib = bibliography("bibliography.bib", style: "harvard-cite-them-right")
#bibliography-slide(bib)

// Appendix slides if needed
#title-slide[
  Appendix
]

#slide(title: "Spatial Patterns in Detail")[
  #set text(size: 18pt)
  #strong[Municipal-level patterns of new different-type neighbors (1985-2020)]

  //#image("https://source.unsplash.com/random/800x500/?denmark+map", width: 80%)
  #text(size: 10pt)[Placeholder - replace with Figure 3 from thesis]

  #v(0.5em)
  - Dark blue/purple represents "low" intensity of new different-type neighbors
  - Orange/yellow represents "high" intensity
  - East-west divide clearly visible
]

#slide(title: "Same-Type Neighbor Trends over Time")[
  #set text(size: 18pt)
  #strong[Evolution of residential sorting (1990-2020)]

  //#image("https://source.unsplash.com/random/800x500/?graph+trend", width: 80%)
  #text(size: 10pt)[Placeholder - replace with Figure 5 from thesis]

  #v(0.5em)
  - Native households: Increasing proportion with exclusively native nearest neighbors
  - By 2020, ~60% of native households had 80-100% same-type neighbors (vs. ~40% in 1990)
  - Non-Western households: Opposite trend suggesting integration, not segregation
  - Counterfactual simulation shows increased segregation beyond what would be expected from demographic changes alone
]

#slide(title: "Robustness Checks")[
  #set text(size: 18pt)
  #strong[Alternative specifications:]

  - Combining all control distances into a single category
  - Varying distance thresholds for nearest neighbors
  - Different neighborhood definitions

  #v(1em)
  #strong[Results remain consistent across specifications:]
  - Spatial decay of effects provides additional support for Schelling mechanism
  - Moving response decreases monotonically with distance to new different-type neighbors
  - Effects primarily concentrated within 25 meters
]

#slide(title: "The Schelling Model Simulation")[
  #set text(size: 18pt)
  #cols(columns: (1fr, 1fr))[
    #strong[Simple agent-based model:]

    - Agents of two types randomly allocated on grid
    - Agents move if share of different-type neighbors exceeds tolerance threshold
    - Even with modest tolerance thresholds, segregation emerges
  ][
    //#image("https://source.unsplash.com/random/800x500/?simulation+grid", width: 100%)
    #text(size: 10pt)[Placeholder - replace with Figure C.1 from thesis appendix]
  ]

  #v(0.5em)
  #align(center)[
    This visualization demonstrates how small individual preferences
    can lead to significant macro-level segregation patterns
  ]
]
