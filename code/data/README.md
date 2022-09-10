Information about the available data

## annotated_tweets.csv

Manually annotated tweets about flooding events and their relevance and impact.

### Attributes
- **id** (int) : Tweet ID
- **Text mainly in Swedish** (0, 1): Language used is Swedish
- **On topic** (0, 1) : tweets that are on topic (but do not necessarily describe the actual event or impact, e.g. tweets refering to a flooding, but just to comment on climate change): "Det är tragisk att vi observerar fler och fler översvämningar under senaste år" -> 1 ;  "Vi alla minns de stora översvämningar i Gävle under 1980-talet" -> 1; "afgafg afgfgz lflflflf" -> 0, "köpa bostad i Gävle" -> 0
- **Informative/relevant/non-sarcastic** (0, 1) : tweets that are on topic and refer to the actual event, provide information about what is happening or about relevant observations, e.g. översvämningar i Gävle, höga flöden i klarälven, starkregn i Linköping som ledde till översvämningar -> 1;   "Det är tragisk att vi observerar fler och fler översvämningar under senaste år" -> 0 ;  "Vi alla minns de stora översvämningar i Gävle under 1980-talet" -> 0;
- **Contains specific information about IMPACTS** (0, 1) : tweets that provide information or refer to any type of specific impact, with reference to where the impact has occured, what has been impacted or who has was effected. Examples: vägen mellan x och x är översvämmad/man kommer inte fram; vi har vatten i vår källare; räddningstjänsten är framme för att evakuerar personer, ...
