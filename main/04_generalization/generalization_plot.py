from svgutils.compose import *

Figure("581cm", "257cm", 
        Panel(
              SVG("figures/generalization_bem.svg"),
             ),
        Panel(
              SVG("figures/generalization_pinns.svg").scale(1.0),
             ).move(290, 0)
        ).save("figures/generalization.svg")