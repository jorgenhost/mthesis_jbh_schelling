#import "@preview/typslides:1.2.5": *
 
// Project configuration
#show: typslides.with(
  ratio: "16-9",
  theme: "dusky",
)

// The front slide is the first slide of your presentation
#front-slide(
  title: "Love thy neighbor?",
  subtitle: [An empirical test of neighborhood change and Schelling behavior],
  authors: "Jørgen Baun Høst",
  info: [#link("https://github.com/jorgenhost/mthesis_jbh_schelling")],
)

// Custom outline
#table-of-contents()

// Title slides create new sections
#title-slide[
  This is a _Title slide_
]

// A simple slide
#slide[
  #framed[Does the ethnicity of your nearest neighbor affect your propensity  to move?]
]


// Slide with title
#slide(title: "This is the slide title")[
  #grayed($ Y_(i t) = beta_1 bb(I) [e', k=n_(n e a r e s t)] + beta_2 bb(I)[e', k=n_(n e a r)] + beta_3 bb(I) [e', k = n_(c l o s e)]+omega_(j, t) + epsilon_(i,j,t)$, text-size: 18pt)

  Sample references: @schelling1971dynamic, @Bayer_2022_nearest_neighbor.
  
]

// - Add your #stress[bibliography slide]!

//     1. `#let bib = bibliography("you_bibliography_file.bib")`
//     2. `#bibliography-slide(bib)`


// Focus slide
#focus-slide[
  This is an auto-resized _focus slide_.
]

// Blank slide
#blank-slide[
  - This is a `#blank-slide`.

  - Available #stress[themes]#footnote[Use them as *color* functions! e.g., `#reddy("your text")`]:

  #framed(back-color: white)[
    #bluey("bluey"), #reddy("reddy"), #greeny("greeny"), #yelly("yelly"), #purply("purply"), #dusky("dusky"), darky.
  ]

  ```typst
  #show: typslides.with(
    ratio: "16-9",
    theme: "bluey",
  )
  ```
]

// Columns
#slide(title: "Columns")[

  #cols(columns: (2fr, 1fr, 2fr), gutter: 2em)[
    #grayed[Columns can be included using `#cols[...][...]`]
  ][
    #grayed[And this is]
  ][
    #grayed[an example.]
  ]

  - Custom spacing: `#cols(columns: (2fr, 1fr, 2fr), gutter: 2em)[...]`
]


// Bibliography
#let bib = bibliography("bibliography.bib", style: "harvard-cite-them-right")
#bibliography-slide(bib)
