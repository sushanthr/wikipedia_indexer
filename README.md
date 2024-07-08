Wikipedia indexer uses an 8 bit quantized version of https://huggingface.co/intfloat/e5-small-v2 to generate embeddings and search a small wikipedia dataset.
The data set used here is https://www.kaggle.com/datasets/jacksoncrow/wikipedia-multimodal-dataset-of-good-articles, consisting of 36,476 articles.

## Usage
Download the archive from https://www.kaggle.com/datasets/jacksoncrow/wikipedia-multimodal-dataset-of-good-articles, and unzip it to a sub-folder archive.
Either use the onnx model from https://huggingface.co/intfloat/e5-small-v2 or quantize it to 8 bits using https://github.com/nixiesearch/onnx-convert
Prequantized version is available here https://github.com/nixiesearch/onnx-convert.

Generate the index by running WikipediaIndexer.py. Note: This takes several hours.
Run WikipediaRetriever.py to query the index.

## Example Q&A

```
Enter your question (or 'quit' to exit): American independence day ?

Top 5 relevant documents:

1. Title: Independence Day (India)
   URL: https://en.wikipedia.org/wiki/Independence_Day_%28India%29
   Text snippet: Independence Day is annually celebrated on 15 August, as a national holiday in India commemorating the nation's independence from the United Kingdom on 15 August 1947, the day when the UK Parliament p...

2. Title: United States
   URL: https://en.wikipedia.org/wiki/United_States
   Text snippet: that all men are created equal and endowed by their Creator with unalienable rights and that those rights were not being protected by Great Britain, and declared, in the words of the resolution, that ...

3. Title: Independence Day (Pakistan)
   URL: https://en.wikipedia.org/wiki/Independence_Day_%28Pakistan%29
   Text snippet: Independence Day (; Yāum-e-Āzādi), observed annually on 14 August, is a national holiday in Pakistan. It commemorates the day when Pakistan achieved independence and was declared a sovereign state fol...

4. Title: Phallic architecture
   URL: https://en.wikipedia.org/wiki/Phallic_architecture
   Text snippet: *...

5. Title: United States Declaration of Independence
   URL: https://en.wikipedia.org/wiki/United_States_Declaration_of_Independence
   Text snippet: The United States Declaration of Independence is the pronouncement adopted by the Second Continental Congress meeting at the Pennsylvania State House (now known as Independence Hall) in Philadelphia, ...
```

```
Enter your question (or 'quit' to exit): Top Hollywood movie ?
Top 5 relevant documents:

1. Title: Hollywood Undercover
   URL: https://en.wikipedia.org/wiki/Hollywood_Undercover
   Text snippet: Category:Hollywood history and culture...

2. Title: Quentin Tarantino
   URL: https://en.wikipedia.org/wiki/Quentin_Tarantino
   Text snippet: Actor/actress ! Reservoir Dogs (1992) ! Pulp Fiction (1994) ! Jackie Brown (1997) ! Kill Bill: Volume 1 (2003) ! Kill Bill: Volume 2 (2004) ! Death Proof (2007) ! Inglourious Basterds (2009) ! Django ...

3. Title: United States
   URL: https://en.wikipedia.org/wiki/United_States
   Text snippet: Kanye West, and Ariana Grande. * * Cinema thumb|The Hollywood Sign in Los Angeles, California|alt=The Hollywood Sign Hollywood, a northern district of Los Angeles, California, is one of the leaders in...

4. Title: A.I. Artificial Intelligence
   URL: https://en.wikipedia.org/wiki/A.I._Artificial_Intelligence
   Text snippet: Category:Stanley Kubrick Category:Warner Bros. films...

5. Title: Batman Begins
   URL: https://en.wikipedia.org/wiki/Batman_Begins
   Text snippet: Category:Superhero thriller films Category:Syncopy Inc. films *1 Category:Warner Bros. films...
```

```
Enter your question (or 'quit' to exit): **President of the united states ?**

Top 5 relevant documents:

1. Title: Bill Clinton
   URL: https://en.wikipedia.org/wiki/Bill_Clinton
   Text snippet: Gun control policy of the Clinton Administration Historical rankings of Presidents of the United States List of Governors of Arkansas List of Presidents of the United States List of Presidents of the ...

2. Title: Chester A. Arthur
   URL: https://en.wikipedia.org/wiki/Chester_A._Arthur
   Text snippet: List of Presidents of the United States List of Presidents of the United States by previous experience Arthur Cottage, ancestral home, Cullybackey, County Antrim, Northern Ireland Julia Sand Notes Ref...

3. Title: Bill Clinton
   URL: https://en.wikipedia.org/wiki/Bill_Clinton
   Text snippet: from C-SPAN's American Presidents: Life Portraits, December 20, 1999 Clinton an American Experience documentary * Category:1946 births Category:2016 United States presidential electors Category:20th-c...

4. Title: W. Webber Kelly
   URL: https://en.wikipedia.org/wiki/W._Webber_Kelly
   Text snippet: Category:Green Bay Packers presidents...

5. Title: Forbidden City
   URL: https://en.wikipedia.org/wiki/Forbidden_City
   Text snippet: President of the United States Donald Trump was the first US President to be granted a state dinner in the Forbidden City since the founding of the People's Republic of China. Structure thumb|left|The...
```

```
Enter your question (or 'quit' to exit): Show me an article about space flight 

Top 5 relevant documents:

1. Title: Apollo 15 postal covers incident
   URL: https://en.wikipedia.org/wiki/Apollo_15_postal_covers_incident
   Text snippet: Additional numbers following page numbers for some books are Kindle locations. Sources External links NASA News Release 72-189, "Articles Carried on Manned Space Flights" from collectspace. com Catego...

2. Title: De Havilland Comet
   URL: https://en.wikipedia.org/wiki/De_Havilland_Comet
   Text snippet: Marlborough, Wiltshire, UK: Crowood Press, 2005. . Davies, R. E. G. and Philip J. Birtles. Comet: The World's First Jet Airliner. McLean, Virginia: Paladwr Press, 1999. . Dennies, Daniel P. How to Org...

3. Title: North American XB-70 Valkyrie
   URL: https://en.wikipedia.org/wiki/North_American_XB-70_Valkyrie
   Text snippet: New York: Prentice Hall, 1986. . Taube, L. J. , Study Manager. "SD 72-SH-0003, B-70 Aircraft Study Final Report, Vol. I". North American Rockwell via NASA, April 1972: Vol. II: Vol. III: Vol. IV. von ...

4. Title: Boeing 747
   URL: https://en.wikipedia.org/wiki/Boeing_747
   Text snippet: Air Transportation: 1903–2003. Dubuque, IA: Kendall Hunt Publishing Co. , 2004. . Lawrence, Philip K. and David Weldon Thornton. Deep Stall: The Turbulent Story of Boeing Commercial Airplanes. Burling...

5. Title: Buzz Aldrin
   URL: https://en.wikipedia.org/wiki/Buzz_Aldrin
   Text snippet: "Satellite of solitude" by Buzz Aldrin: an article in which Aldrin describes what it was like to walk on the Moon, Cosmos science magazine Category:1930 births Category:Living people Category:1966 in ...
```

```
Enter your question (or 'quit' to exit): What is the tallest mountain in Washington ?

Top 5 relevant documents:

1. Title: Glacier Peak
   URL: https://en.wikipedia.org/wiki/Glacier_Peak
   Text snippet: New Westminster and Port Coquitlam. The volcano is the fourth tallest peak in Washington state, and not as much is known about it compared to other volcanoes in the area. Local Native Americans have r...

2. Title: Mount Rainier
   URL: https://en.wikipedia.org/wiki/Mount_Rainier
   Text snippet: thumb|Mount Rainier, as viewed from Kerry Park in Seattle thumb|Mount Rainier from an aircraft Mount Rainier is the tallest mountain in Washington and the Cascade Range. This peak is located just east...

3. Title: Mount Rainier
   URL: https://en.wikipedia.org/wiki/Mount_Rainier
   Text snippet: thumb|Mt Rainier from ISS 2018 Mount Rainier (pronounced: ), also known as Tahoma or Tacoma, is a large active stratovolcano in Cascadia located  south-southeast of Seattle, in Mount Rainier National ...

4. Title: Mount Baker
   URL: https://en.wikipedia.org/wiki/Mount_Baker
   Text snippet: Mount Baker is the most heavily glaciated of the Cascade Range volcanoes; the volume of snow and ice on Mount Baker, is greater than that of all the other Cascades volcanoes (except Rainier) combined....

5. Title: Mount Washington (New Hampshire)
   URL: https://en.wikipedia.org/wiki/Mount_Washington_%28New_Hampshire%29
   Text snippet: due to harsh and rapidly changing conditions, inadequate equipment, and failure to plan for the wide variety of conditions which can occur above tree line. The weather at Mount Washington has made it ...
```

```
Enter your question (or 'quit' to exit):How many calories are in a burger ?

Top 5 relevant documents:

1. Title: Whopper
   URL: https://en.wikipedia.org/wiki/Whopper
   Text snippet: The whole burger contains more than the recommended daily allowance of calories for men at 2, 520 calories, with 144 grams of fat, 59g of which is saturated, and 3, 780 mg of sodium, more than double ...

2. Title: Burger King legal issues
   URL: https://en.wikipedia.org/wiki/Burger_King_legal_issues
   Text snippet: According to the statement by the company's corporate parent, Burger King Brands, the meals will contain no more than 560 calories per meal, with less than 30 percent of the calories derived from fat,...

3. Title: Smashburger
   URL: https://en.wikipedia.org/wiki/Smashburger
   Text snippet: such as in Kalamazoo and Grand Rapids restaurants, where the local burger is topped with olives, a tradition that had once been locally popular. The restaurant attracted criticism from health experts ...

4. Title: Burger King legal issues
   URL: https://en.wikipedia.org/wiki/Burger_King_legal_issues
   Text snippet: PETA and the Humane Society of the United States were quoted as saying that Burger King’s initiatives put it ahead of its competitors in terms of animal rights and welfare and that they were hopeful t...

5. Title: Burger King grilled chicken sandwiches
   URL: https://en.wikipedia.org/wiki/Burger_King_grilled_chicken_sandwiches
   Text snippet: At the time, the sandwich had 379 calories and 18 grams of fat, 10 of which came from the sauce. The introduction of the BK Broiler was one of the most successful restaurant product launches ever, enc...
Enter your question (or 'quit' to exit):
```

