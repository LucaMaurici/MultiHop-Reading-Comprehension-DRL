def indexer(docNumber, sentenceIndex):
    return 'd'+str(docNumber) + 's'+str(sentenceIndex)

def getRadixNode(node):
    dPos = node.index('s')
    docNumber = node[1:dPos]
    print(indexer(docNumber, 0))
    return indexer(docNumber, 0)

def shareEntities(list1, list2):
    for e1 in list1:
        if e1 in list2:
            return True
    return False

def shareWords(sentence1, sentence2):
    list1 = sentence1.lower().split()
    list2 = sentence2.lower().split()
    for e1 in list1:
        if e1 in list2:
            return True
    return False

def buildCoreferenceGraph(question, documents):

    from pynlp import StanfordCoreNLP
    #from nltk import tokenize
    #import nltk
    #nltk.download('punkt')
    import Graph_class as gs
    import pickle

    annotators = 'tokenize, ssplit, pos, lemma, ner, entitymentions, coref, sentiment, openie'
    #annotators = ''
    options = {'openie.resolve_coref': True}

    nlp = StanfordCoreNLP(annotators=annotators, options=options)


    '''
    documents = [
       "Canada (French: ) is a country in the northern half of North America. Its ten provinces and three territories extend from the Atlantic to the Pacific and northward into the Arctic Ocean, covering , making it the world's second-largest country by total area and the fourth-largest country by land area. Canada's border with the United States is the world's longest land border. The majority of the country has a cold or severely cold winter climate, but southerly areas are warm in summer. Canada is sparsely populated, the majority of its land territory being dominated by forest and tundra and the Rocky Mountains. About four-fifths of the country's population of 36 million people is urbanized and live near the southern border. Its capital is Ottawa, its largest city is Toronto; other major urban areas include Montreal, Vancouver, Calgary, Edmonton, Quebec City, Winnipeg and Hamilton.",
       "Origins of the Charlton Press.\nJames (Jim) Charlton (July 26, 1911 - September 20, 2013) It was Jim Charltons experience as an avid numismatist that inspired him to create a catalogue for coin collectors. In 1926, when Jim Charlton was 15, his older brother Harry Charlton gifted him a United States 1863 Indian Head cent.",
       "The Indian Head cent, also known as an Indian Head penny, was a one-cent coin ($0.01) produced by the United States Bureau of the Mint from 1859 to 1909. It was designed by James Barton Longacre, the Chief Engraver at the Philadelphia Mint.",
       "James E. Charlton ( July 26 , 1911 -- September 20 , 2013 ) was a Canadian coin dealer and numismatic publisher . After working as a stationary engineer , he opened a coin store in Toronto . He published his first guidebook , Catalogue of Canadian Coins , Tokens & Fractional Currency , in 1952 . Other titles from Charlton Press followed . Subsequently he sold his store to concentrate on his publications , and later sold his publishing company , Charlton Press , which continues to publish price guides for coins , banknotes and other collectibles . He turned 100 in July 2011 .",
       "Los Angeles (Spanish for \"The Angels\"), officially the City of Los Angeles and often known by its initials L.A., is the second-most populous city in the United States (after New York City), the most populous city in California and the county seat of Los Angeles County. Situated in Southern California, Los Angeles is known for its mediterranean climate, ethnic diversity, sprawling metropolis, and as a major center of the American entertainment industry. Los Angeles lies in a large coastal basin surrounded on three sides by mountains reaching up to and over .",
       "A coin is a small, flat, round piece of metal or plastic used primarily as a medium of exchange or legal tender. They are standardized in weight, and produced in large quantities at a mint in order to facilitate trade. They are most often issued by a government.",
       "Mexico City, officially City of Mexico (, ; abbreviated as \"CDMX\"), is the capital and most populous city of the United Mexican States. As an \"alpha\" global city, Mexico City is one of the most important financial centers in the Americas. It is located in the Valley of Mexico (\"Valle de M\u00e9xico\"), a large valley in the high plateaus at the center of Mexico, at an altitude of . The city consists of sixteen municipalities (previously called boroughs).",
       "A global city, also called world city or sometimes alpha city or world center, is a city generally considered to be an important node in the global economic system. The concept comes from geography and urban studies, and the idea that globalization can be understood as largely created, facilitated, and enacted in strategic geographic locales according to a hierarchy of importance to the operation of the global system of finance and trade.",
       "The Greater Toronto Area (GTA) is the most populous metropolitan area in Canada. At the 2011 census, it had a population of 6,054,191, and the census metropolitan area had a population of 5,583,064. The Greater Toronto Area is defined as the central city of Toronto, and the four regional municipalities that surround it: Durham, Halton, Peel, and York. The regional span of the Greater Toronto Area is sometimes combined with the city of Hamilton, Ontario and its surrounding region, to form the Greater Toronto and Hamilton Area. The Greater Toronto Area is the northern part of the Golden Horseshoe.",
       "Chicago (or ), officially the City of Chicago, is the third-most populous city in the United States, and the fifth-most populous city in North America. With over 2.7\u00a0million residents, it is the most populous city in the state of Illinois and the Midwestern United States, and the county seat of Cook County. The Chicago metropolitan area, often referred to as Chicagoland, has nearly 10\u00a0million people and is the third-largest in the U.S.",
       "Toronto is the most populous city in Canada, the provincial capital of Ontario, and the centre of the Greater Toronto Area, the most populous metropolitan area in Canada. Growing in population, the 2011 census recorded a population of 2,615,060. As of 2015, the population is now estimated at 2,826,498, making Toronto the fourth-largest city in North America based on the population within its city limits. Toronto trails only Mexico City, New York City, and Los Angeles by this measure, while it is the fifth-largest (behind also Chicago) if ranked by the size of its metropolitan area . An established global city, Toronto is an international centre of business, finance, arts, and culture, and widely recognized as one of the most multicultural and cosmopolitan cities in the world.",
       "The City of New York, often called New York City or simply New York, is the most populous city in the United States. With an estimated 2015 population of 8,550,405 distributed over a land area of just , New York City is also the most densely populated major city in the United States. Located at the southern tip of the state of New York, the city is the center of the New York metropolitan area, one of the most populous urban agglomerations in the world. A global power city, New York City exerts a significant impact upon commerce, finance, media, art, fashion, research, technology, education, and entertainment, its fast pace defining the term \"New York minute\". Home to the headquarters of the United Nations, New York is an important center for international diplomacy and has been described as the cultural and financial capital of the world.",
       "Ontario, one of the 13 provinces and territories of Canada, is located in east-central Canada. It is Canada's most populous province by a large margin, accounting for nearly 40 percent of all Canadians, and is the second-largest province in total area. Ontario is fourth-largest in total area when the territories of the Northwest Territories and Nunavut are included. It is home to the nation's capital city, Ottawa, and the nation's most populous city, Toronto."
       ]
    '''

    '''
    documents = [
        "The Cascade Range or Cascades is a major mountain range of western North America, extending from southern British Columbia through Washington and Oregon to Northern California. It includes both non-volcanic mountains, such as the North Cascades, and the notable volcanoes known as the High Cascades. The small part of the range in British Columbia is referred to as the Canadian Cascades or, locally, as the Cascade Mountains. The latter term is also sometimes used by Washington residents to refer to the Washington section of the Cascades in addition to North Cascades, the more usual U.S. term, as in North Cascades National Park. The highest peak in the range is Mount Rainier in Washington at .",
        "The Okanogan National Forest is a U.S. National Forest located in Okanogan County in north-central Washington, United States.",
        "The United States Forest Service (USFS) is an agency of the U.S. Department of Agriculture that administers the nation's 154 national forests and 20 national grasslands, which encompass . Major divisions of the agency include the National Forest System, State and Private Forestry, Business Operations, and the Research and Development branch. Managing approximately 25% of federal lands, it is the only major national land agency that is outside the U.S. Department of the Interior.",
        "Mount Rainier National Park is a United States National Park located in southeast Pierce County and northeast Lewis County in Washington state. It was established on March 2, 1899 as the fifth national park in the United States. The park encompasses including all of Mount Rainier, a stratovolcano. The mountain rises abruptly from the surrounding land with elevations in the park ranging from 1,600 feet to over 14,000 feet (490 - 4,300\u00a0m). The highest point in the Cascade Range, around it are valleys, waterfalls, subalpine meadows, old-growth forest and more than 25 glaciers. The volcano is often shrouded in clouds that dump enormous amounts of rain and snow on the peak every year.",
        "Snoqualmie National Forest is a United States National Forest in the State of Washington. It was established on 1 July 1908, when an area of 961,120 acres (3,889.52 km\u00b2) was split from the existing Washington National Forest. Its size was increased on 13 October 1933, when a part of Rainier National Forest was added. In 1974 Snoqualmie was administratively combined with Mount Baker National Forest to make Mount Baker-Snoqualmie National Forest. In descending order of land area, Snoqualmie National Forest lies in parts of King, Snohomish, Pierce, and Kittitas counties. There are local ranger district offices in North Bend and Skykomish. Its main base is in Everett, Washington. As of 30 September 2007, it had an area of 1,258,167 acres (5,091.62 km\u00b2), representing about 49 percent of the combined forest's total acreage.",
        "Chelan National Forest was established in Washington by the U.S. Forest Service on July 1, 1908 with from a portion of Washington National Forest. On July 1, 1921 it absorbed the first Okanogan National Forest, but on March 23, 1955 the name was changed back to Okanogan.",
        "Mount Baker (Lummi: '; or '), also known as Koma Kulshan or simply Kulshan, is an active glaciated andesitic stratovolcano in the Cascade Volcanic Arc and the North Cascades of Washington in the United States. Mount Baker has the second-most thermally active crater in the Cascade Range after Mount Saint Helens. About due east of the city of Bellingham, Whatcom County, Mount Baker is the youngest volcano in the Mount Baker volcanic field. While volcanism has persisted here for some 1.5 million years, the current glaciated cone is likely no more than 140,000 years old, and possibly no older than 8090,000 years. Older volcanic edifices have mostly eroded away due to glaciation.",
        "Mount Baker National Forest was established in Washington on January 21, 1924 when its name was changed from Washington National Forest. In 1974 it was administratively combined with Snoqualmie National Forest to make Mount Baker-Snoqualmie National Forest. In descending order of land area, Mount Baker National Forest is located in parts of Snohomish, Whatcom, and Skagit counties. As of 30 September 2007, it had an area of 1,301,787 acres (5,268.1\u00a0km\u00b2), representing about 51 percent of the combined forest's area. There are local ranger district offices located in Darrington and Sedro-Woolley.",
        "The Public Land Survey System (PLSS) is the surveying method developed and used in the United States to plat, or divide, real property for sale and settling. Also known as the Rectangular Survey System, it was created by the Land Ordinance of 1785 to survey land ceded to the United States by the Treaty of Paris in 1783, following the end of the American Revolution. Beginning with the Seven Ranges, in present-day Ohio, the PLSS has been used as the primary survey method in the United States. Following the passage of the Northwest Ordinance, in 1787, the Surveyor General of the Northwest Territory platted lands in the Northwest Territory. The Surveyor General was later merged with the General Land Office, which later became a part of the U.S Bureau of Land Management, or BLM. Today, the BLM controls the survey, sale, and settling of the new lands, and manages the State Plane Coordinate System.",
        "The Land Ordinance of 1785 was adopted by the United States Congress of the Confederation on May 20, 1785. It set up a standardized system whereby settlers could purchase title to farmland in the undeveloped west. Congress at the time did not have the power to raise revenue by direct taxation, so land sales provided an important revenue stream. The Ordinance set up a survey system that eventually covered over three-fourths of the area of the continental United States.",
        "The Mount Baker-Snoqualmie National Forest in Washington is a National Forest extending more than along the western slopes of the Cascade Range from the CanadaUS border to the northern boundary of Mount Rainier National Park. Administered by the United States Forest Service, the forest is headquartered in Everett.",
        "The Department of the Treasury is an executive department and the treasury of the United States federal government. It was established by an Act of Congress in 1789 to manage government revenue. The Department is administered by the Secretary of the Treasury, who is a member of the Cabinet. Jacob J. Lew is the current Secretary of the Treasury; he was sworn in on February 28, 2013.",
        "The General Land Office (GLO) was an independent agency of the United States government responsible for public domain lands in the United States. It was created in 1812 to take over functions previously conducted by the United States Department of the Treasury. Starting with the passage of The Land Ordinance of 1785, which created the Public Land Survey System, the Treasury Department had already overseen the survey of the \"Northwest Territory\" including what is now the State of Ohio.",
        "Washington National Forest was established by the General Land Office as the Washington Forest Reserve in Washington on February 22 , 1897 with 3,594,240 acres ( 14,545.4 km2 ) . After the transfer of federal forests to the U.S. Forest Service in 1905 , it became a National Forest on March 4 , 1907 . On July 1 , 1908 , Chelan National Forest was established with a portion of Washington . On January 21 , 1924 Washington was renamed Mount Baker National Forest . The lands presently exist as Mount Baker - Snoqualmie National Forest ."
    ]
    '''

    '''
    question = 'publication_date blueprint '
    documents = [
        "McKinley Morganfield (April 4, 1913  April 30, 1983), better known as Muddy Waters, was an American blues musician who is often cited as the \"father of modern Chicago blues\".",
        "The guitar is a musical instrument classified as a string instrument with anywhere from four to 18 strings, usually having six. The sound is projected either acoustically, using a hollow wooden or plastic and wood box (for an acoustic guitar), or through electrical amplifier and a speaker (for an electric guitar). It is typically played by strumming or plucking the strings with the fingers, thumb and/or fingernails of the right hand or with a pick while fretting (or pressing against the frets) the strings with the fingers of the left hand. The guitar is a type of chordophone, traditionally constructed from wood and strung with either gut, nylon or steel strings and distinguished from other chordophones by its construction and tuning. The modern guitar was preceded by the gittern, the vihuela, the four-course Renaissance guitar, and the five-course baroque guitar, all of which contributed to the development of the modern six-string instrument.",
        "Blueprint is the third album by Irish guitarist Rory Gallagher , released as a vinyl record in 1973 . With his first band Taste and with his solo band up to this point Gallagher was one of the first guitarists to lead a power trio lineup . With Blueprint Gallagher included a keyboardist for the first time .",
        "William Rory Gallagher (; 2 March 1948\u00a0 14 June 1995) was an Irish blues and rock multi-instrumentalist, songwriter, and bandleader. Born in Ballyshannon, County Donegal, and brought up in Cork, Gallagher recorded solo albums throughout the 1970s and 1980s, after forming the band Taste during the late 1960s. He was a talented guitarist known for his charismatic performances and dedication to his craft. Gallagher's albums have sold in excess of 30 million copies worldwide. Gallagher received a liver transplant in 1995, but died of complications later that year in London, UK at the age of 47.",
        "The Hammond organ is an electric organ, invented by Laurens Hammond and John M. Hanert and first manufactured in 1935. Various models have been produced, most of which use sliding drawbars to create a variety of sounds. Until 1975, Hammond organs generated sound by creating an electric current from rotating a metal tonewheel near an electromagnetic pickup, and then strengthening the signal with an amplifier so that it can drive a speaker cabinet. Around two million Hammond organs have been manufactured. The organ is commonly used with, and associated with, the Leslie speaker.",
        "County Donegal (pronounced or ) is a county of Ireland. It is part of the Border Region of the Republic of Ireland and is in the province of Ulster. It is named after the town of Donegal in the south of the county. Donegal County Council is the local council for the county and Lifford serves as the county town. The population of the county is 158,755 according to the 2016 census. It has also been known as (County) Tyrconnell (\"\"), after the historic territory of the same name.",
        "A power trio is a rock and roll band format having a lineup of guitar, bass and drums, leaving out the second guitar or keyboard that are used in other rock music quartets and quintets to fill out the sound with chords. While one or more band members typically sing, power trios emphasize instrumental performance and overall impact over vocals and lyrics.\nHistory.\nThe rise of the power trio in the 1960s was made possible in part by developments in amplifier technology that greatly enhanced the volume of the electric guitar and bass. Particularly, the popularization of the electric bass guitar defined the bottom end and filled in the gaps. Since the amplified bass could also now be louder, the rest of the band could also play at higher volumes, without fear of being unable to hear the bass. This allowed a three-person band to have the same sonic impact as a large band but left far more room for improvisation and creativity, unencumbered by the need for detailed arrangements. As with the organ trio, a 1960s-era soul jazz group centered on the amplified Hammond organ, a three-piece group could fill a large bar or club with a big sound for a much lower price than a large rock and roll band. A power trio, at least in its blues rock incarnation, is also generally held to have developed out of Chicago-style blues bands such as Muddy Waters' trio."
    ]
    '''

    '''
    question = 'producer the snapper '
    documents = [
        "The Snapper is a 1993 Irish television film which was directed by Stephen Frears and starred Tina Kellegher , Colm Meaney and Brendan Gleeson . The film is based on the novel by Irish writer Roddy Doyle , about the Rabbitte family and their domestic adventures .",
        "My Beautiful Laundrette is a 1985 British comedy-drama film directed by Stephen Frears from a screenplay by Hanif Kureishi. The film was also one of the first films released by Working Title Films.",
        "BBC Radio 4 is a radio station owned and operated by the British Broadcasting Corporation (BBC) that broadcasts a wide variety of spoken-word programmes including news, drama, comedy, science and history. It replaced the BBC Home Service in 1967. The station controller is Gwyneth Williams; and the station is part of BBC Radio and the \"BBC Radio\" department. The station is broadcast from the BBC's headquarters at Broadcasting House, London.",
        "Roddy Doyle (8 May 1958 is an Irish novelist, dramatist and screenwriter. He was the author of ten novels for adults, eight books for children, seven plays and screenplays, and dozens of short stories. Several of his books have been made into films, beginning with \"The Commitments\" in 1991. Doyle's work is set primarily in Ireland, especially working-class Dublin, and is notable for its heavy use of dialogue written in slang and Irish English dialect. Doyle was awarded the Booker Prize in 1993 for his novel \"Paddy Clarke Ha Ha Ha\".",
        "Stephen Arthur Frears (born 20 June 1941) is an English film director. Frears has directed British films since the 1980s including \"My Beautiful Laundrette\", \"Dangerous Liaisons\", \"High Fidelity\", \"The Queen\" and \"Philomena\". He has been nominated for two Academy Awards for Best Director for \"The Grifters\" and \"The Queen\".",
        "Tina Kellegher (born 1967 in Dublin), is an Irish actress, best known for her role as Niamh Quigley in BBC television series \"Ballykissangel\". She is also a well-known voice on BBC Radio 4, having played policewoman Tina Mahon in the first four series of \"Baldi\". She is married to Gordon Wycherley, location manager on, among other projects, \"Ballykissangel\". They have two sons, Michael, who was born in 2003, and Brian, who was born in 2007. Her brother-in-law is fellow \"Ballykissangel\" star Don Wycherley.",
        "Colm J. Meaney (Irish: \"Colm \u00d3 Maonaigh\"; born 30 May 1953) is an Irish actor known for playing Miles O'Brien in \"\" and \"\". He has guest-starred on many TV shows from \"Law & Order\" to \"The Simpsons\", and during its run, starred as railroad magnate Thomas Durant on AMC's drama series \"Hell on Wheels\".",
        "The British Broadcasting Corporation (BBC) is a British public service broadcaster. It is headquartered at Broadcasting House in London, is the world's oldest national broadcasting organisation, and is the largest broadcaster in the world by number of employees, with over 20,950 staff in total, of whom 16,672 are in public sector broadcasting; including part-time, flexible as well as fixed contract staff, the total number is 35,402.",
        "Ballykissangel is a BBC television drama created by Kieran Prendiville and set in Ireland, produced in-house by BBC Northern Ireland. The original story revolved around a young English Roman Catholic priest as he became part of a rural community. It ran for six series, which were first broadcast on BBC One in the United Kingdom from 1996 to 2001. It aired in Ireland on RT\u00c9 One and in Australia on ABC TV from 1996 to 2001. Reruns have been aired on Drama in the United Kingdom and in the United States on some PBS affiliates.",
        "The Simpsons is an American animated sitcom created by Matt Groening for the Fox Broadcasting Company. The series is a satirical depiction of working-class life epitomized by the Simpson family, which consists of Homer, Marge, Bart, Lisa, and Maggie. The show is set in the fictional town of Springfield and parodies American culture, society, television, and the human condition.",
        "The Man Booker Prize for Fiction (formerly known as the Booker-McConnell Prize and commonly known simply as the Booker Prize) is a literary prize awarded each year for the best original novel, written in the English language and published in the UK. The winner of the Man Booker Prize is generally assured of international renown and success; therefore, the prize is of great significance for the book trade. From its inception, only Commonwealth, Irish, and South African (later Zimbabwean) citizens were eligible to receive the prize; in 2014, however, this eligibility was widened to any English-language novel."
    ]
    '''

    '''
    question = 'residence ali abdullah ahmed'
    documents = [
        "Hispaniola (Spanish: \"La Espa\u00f1ola\"; Latin: \"Hispaniola\"; Ta\u00edno: \"Haiti\") is the 22nd-largest island in the world, located in the Caribbean island group, the Greater Antilles. It is the second largest island in the Caribbean after Cuba, and the tenth most populous island in the world.",
        "An Internment Serial Number (ISN) is an identification number assigned to captives who come under control of the United States Department of Defense (DoD) during armed conflicts.",
        "Yemen ('), officially known as the Republic of Yemen ('), is an Arab country in Western Asia, occupying South Arabia, the southern end of the Arabian Peninsula. Yemen is the second-largest country in the peninsula, occupying 527,970\u00a0km (203,850\u00a0sq\u00a0mi). The coastline stretches for about 2,000\u00a0km (1,200\u00a0mi). It is bordered by Saudi Arabia to the north, the Red Sea to the west, the Gulf of Aden and Arabian Sea to the south, and Oman to the east-northeast. Although Yemen's constitutionally stated capital is the city of Sana'a, the city has been under rebel control since February 2015. Because of this, Yemen's capital has been temporarily relocated to the port city of Aden, on the southern coast. Yemen's territory includes more than 200 islands; the largest of these is Socotra.",
        "Isla de la Juventud is the second-largest Cuban island and the seventh-largest island in the West Indies (after Cuba itself, Hispaniola, Jamaica, Puerto Rico, Trinidad, and Andros Island). The island was called the Isle of Pines (Isla de Pinos) until 1978. It has an area and is south of the island of Cuba, across the Gulf of Bataban\u00f3. The island lies almost directly south of Havana and Pinar del R\u00edo and is a Special Municipality, not part of any province and is therefore administered directly by the central government of Cuba. The island has only one municipality, also named Isla de la Juventud.",
        "Camag\u00fcey is a city and municipality in central Cuba and is the nation's third largest city with more than 321,000 inhabitants. It is the capital of the Camag\u00fcey Province.",
        "Florida (Spanish for \"land of flowers\") is a state located in the southeastern region of the United States. It is bordered to the west by the Gulf of Mexico, to the north by Alabama and Georgia, to the east by the Atlantic Ocean, and to the south by the Straits of Florida and Cuba. Florida is the 22nd most extensive, the 3rd most populous, and the 8th most densely populated of the U.S. states. Jacksonville is the most populous municipality in the state and is the largest city by area in the contiguous United States. The Miami metropolitan area is Florida's most populous urban area. The city of Tallahassee is the state capital.",
        "Cuba, officially the Republic of Cuba, is a country comprising the island of Cuba as well as Isla de la Juventud and several minor archipelagos. Cuba is located in the northern Caribbean where the Caribbean Sea, the Gulf of Mexico, and the Atlantic Ocean meet. It is south of both the U.S. state of Florida and the Bahamas, west of Haiti, and north of Jamaica. Havana is the largest city and capital; other major cities include Santiago de Cuba and Camag\u00fcey. Cuba is the largest island in the Caribbean, with an area of , and the second-most populous after Hispaniola, with over 11 million inhabitants.",
        "Guantanamo Bay Naval Base, also known as Naval Station Guantanamo Bay or NSGB, (also called GTMO because of the airfield designation code or Gitmo because of the common pronunciation of this code by the U.S. military) is a United States military base located on of land and water at Guant\u00e1namo Bay, Cuba, which the US leased for use as a coaling and naval station in 1903 (for $2,000 per year until 1934, when it was increased to $4,085 per year). The base is on the shore of Guant\u00e1namo Bay at the southeastern end of Cuba. It is the oldest overseas U.S. Naval Base. Since the Cuban Revolution of 1959, the Cuban government has consistently protested against the U.S. presence on Cuban soil and called it illegal under international law, alleging that the base was imposed on Cuba by force. At the United Nations Human Rights Council in 2013, Cuba's Foreign Minister demanded the U.S. return the base and the \"usurped territory\", which the Cuban government considers to be occupied since the U.S. invasion of Cuba during the SpanishAmerican War in 1898.",
        "The Guantanamo Bay detention camp is a United States military prison located within Guantanamo Bay Naval Base, also referred to as Guant\u00e1namo or GTMO (pronounced 'gitmo'), which fronts on Guant\u00e1namo Bay in Cuba. Since the inmates have been detained indefinitely without trial and several inmates were severely tortured, this camp is considered as a major breach of human rights by great parts of the world.",
        "The Caribbean (or ) is a region that consists of the Caribbean Sea, its islands (some surrounded by the Caribbean Sea and some bordering both the Caribbean Sea and the North Atlantic Ocean) and the surrounding coasts. The region is southeast of the Gulf of Mexico and the North American mainland, east of Central America, and north of South America.",
        "The Bahamas, known officially as the Commonwealth of the Bahamas, is an archipelagic state within the Lucayan Archipelago. It consists of more than 700 islands, cays, and islets in the Atlantic Ocean and is located north of Cuba and Hispaniola (Haiti and the Dominican Republic); northwest of the Turks and Caicos Islands; southeast of the US state of Florida and east of the Florida Keys. The state capital is Nassau on the island of New Providence. The designation of \"The Bahamas\" can refer to either the country or the larger island chain that it shares with the Turks and Caicos Islands. As stated in the mandate/manifesto of the Royal Bahamas Defence Force, the Bahamas territory encompasses of ocean space.",
        "The Department of Defense (DoD, USDOD, or DOD) is an executive branch department of the federal government of the United States charged with coordinating and supervising all agencies and functions of the government concerned directly with national security and the United States Armed Forces. The Department is the largest employer in the world, with nearly 1.3 million active duty servicemen and women as of 2016. Adding to its employees are over 801,000 National Guardsmen and Reservists from the four services, and over 740,000 civilians bringing the total to over 2.8 million employees. It is headquartered at the Pentagon in Arlington, Virginia, just outside of Washington, D.C.",
        "The Gulf of Aden (\"\") is a gulf located in the Arabian Sea between Yemen, on the south coast of the Arabian Peninsula, and Somalia in the Horn of Africa. In the northwest, it connects with the Red Sea through the Bab-el-Mandeb strait, which is more than 20 miles wide. It shares its name with the port city of Aden in Yemen, which forms the northern shore of the gulf. Historically, the Gulf of Aden was known as \"The Gulf of Berbera\", named after the ancient Somali port city of Berbera on the south side of the gulf. However, as the city of Aden grew during the colonial era, the name of \"Gulf of Aden\" was popularised.",
        "Ali Abdullah Ahmed , also known as Salah Ahmed al - Salami ( Arabic :    ) ( January 12 , 1970 -- June 10 , 2006 ) , was a citizen of Yemen who died while being held as an enemy combatant in the United States Guantanamo Bay detainment camps , in Cuba . His Guantanamo Internment Serial Number was 693 . Joint Task Force Guantanamo counter-terror analysts estimated he was born in 1977 , in Ib , Yemen . Ali Abdullah Ahmed died in custody on June 10 , 2006 . His death was announced by the Department of Defense as a suicide , on the same day that the deaths of two other detainees were said to be suicides . Their deaths received wide coverage in the media . His younger brother , Muhammaed Yasir Ahmed Taher , is also held in Guantanamo . They had been held since 2002 .",
        "The Arabian Sea is a region of the northern Indian Ocean bounded on the north by Pakistan and Iran, on the west by northeastern Somalia and the Arabian Peninsula, and on the east by India. Historically the sea has been known by other names including the Erythraean Sea and the Persian Sea. Its total area is and its maximum depth is . The Gulf of Aden is in the southwest, connecting the Arabian Sea to the Red Sea through the strait of Bab-el-Mandeb, and the Gulf of Oman is in the northwest, connecting it to the Persian Gulf.",
        "Mexico (, modern Nahuatl ), officially the United Mexican States, is a federal republic in the southern half of North America. It is bordered to the north by the United States; to the south and west by the Pacific Ocean; to the southeast by Guatemala, Belize, and the Caribbean Sea; and to the east by the Gulf of Mexico. Covering almost two million square kilometers (over 760,000\u00a0sq\u00a0mi), Mexico is the sixth largest country in the Americas by total area and the 13th largest independent nation in the world. With an estimated population of over 120 million, it is the eleventh most populous country and the most populous Spanish-speaking country in the world while being the second most populous country in Latin America. Mexico is a federation comprising 31 states and a federal district that is also its capital and most populous city. Other metropolises include Guadalajara, Monterrey, Puebla, Toluca, Tijuana and Le\u00f3n.",
        "Oman (; ' ), officially the Sultanate of Oman, is an Arab country on the southeastern coast of the Arabian Peninsula. Holding a strategically important position at the mouth of the Persian Gulf, the nation is bordered by the United Arab Emirates to the northwest, Saudi Arabia to the west, and Yemen to the southwest, and shares marine borders with Iran and Pakistan. The coast is formed by the Arabian Sea on the southeast and the Gulf of Oman on the northeast. The Madha and Musandam exclaves are surrounded by the UAE on their land borders, with the Strait of Hormuz (which it shares with Iran) and Gulf of Oman forming Musandam's coastal boundaries.",
        "South Arabia is a historical region that consists of the southern region of the Arabian Peninsula, mainly centered in what is now the Republic of Yemen, yet it has historically also included Najran, Jizan, and 'Asir, which are presently in Saudi Arabia, and the Dhofar of present-day Oman.",
        "The Caribbean Sea is a sea of the Atlantic Ocean in the tropics of the Western Hemisphere. It is bounded by Mexico and Central America to the west and south west, to the north by the Greater Antilles starting with Cuba, to the east by the Lesser Antilles, and to the south by the north coast of South America.",
        "Saudi Arabia, officially known as the Kingdom of Saudi Arabia (KSA), is an Arab sovereign state in Western Asia constituting the bulk of the Arabian Peninsula. With a land area of approximately , Saudi Arabia is geographically the fifth-largest state in Asia and second-largest state in the Arab world after Algeria. Saudi Arabia is bordered by Jordan and Iraq to the north, Kuwait to the northeast, Qatar, Bahrain, and the United Arab Emirates to the east, Oman to the southeast, and Yemen to the south. It is separated from Israel and Egypt by the Gulf of Aqaba. It is the only nation with both a Red Sea coast and a Persian Gulf coast, and most of its terrain consists of arid desert or barren landforms.",
        "The Gulf of Mexico is an ocean basin largely surrounded by the North American continent. It is bounded on the northeast, north and northwest by the Gulf Coast of the United States, on the southwest and south by Mexico, and on the southeast by Cuba. The U.S. states of Alabama, Florida, Louisiana, Mississippi and Texas border the Gulf on the north, which are often referred to as the \"Third Coast\" in comparison with the U.S. Atlantic and Pacific coasts, or sometimes the \"south coast\", in juxtaposition to the Great Lakes region being the \"north coast.\" One of the gulf's seven main areas is the Gulf of Mexico basin.",
        "Havana (Spanish: \"La Habana\") is the capital city, largest city, province, major port, and leading commercial centre of Cuba. The city proper has a population of 2.1 million inhabitants, and it spans a total of  making it the largest city by area, the most populous city, and the third largest metropolitan area in the Caribbean region. The city extends mostly westward and southward from the bay, which is entered through a narrow inlet and which divides into three main harbours: Marimelena, Guanabacoa and Atar\u00e9s. The sluggish Almendares River traverses the city from south to north, entering the Straits of Florida a few miles west of the bay.",
        "Joint Task Force Guantanamo (JTF-GTMO) is a U.S. military joint task force based at Guantanamo Bay Naval Base, Guant\u00e1namo Bay, Cuba on the southeastern end of the island. JTF-GTMO falls under US Southern Command. Since January 2002 the command has operated the Guantanamo Bay detention camps Camp X-Ray and its successors Camp Delta, Camp V, and Camp Echo, where detained prisoners are held who have been captured in the war in Afghanistan and elsewhere since the September 11, 2001 attacks. The unit is currently under the command of Rear Admiral Peter J. Clarke.",
        "Santiago de Cuba is the second largest city of Cuba and capital city of Santiago de Cuba Province in the south-eastern area of the island, some south-east of the Cuban capital of Havana.",
        "Jamaica is an island country situated in the Caribbean Sea, consisting of the third-largest island of the Greater Antilles. The island, in area, lies about south of Cuba, and west of Hispaniola (the island containing the nation-states of Haiti and the Dominican Republic). Jamaica is the fourth-largest island country in the Caribbean, by area.",
        "Western Asia, West Asia, Southwestern Asia or Southwest Asia is the westernmost subregion of Asia. The concept is in limited use, as it significantly overlaps with the Middle East (or Near East), the main difference being the exclusion of Egypt (which would be counted as part of North Africa). The term is sometimes used for the purposes of grouping countries in statistics."
    ]
    '''

    temp = list()
    for document in documents:
        document = document.replace('%', ' percent')
        #document = document.replace('+', '<plus>')
        temp.append(document)
    documents = temp
    print(documents)

    graph = gs.Graph()

    id2sentence = {}
    id2entities = {}
    #nodes2radix = {}

    graph.addNode('q')
    id2sentence['q'] = question


    for docNumber, text in enumerate(documents, start=0):

        print('\n\n********* STARTING DOCUMENT ', docNumber, ' *********\n\n')

        #text = ("Canada (French: ) is a country in the northern half of North America. Its ten provinces and three territories extend from the Atlantic to the Pacific and northward into the Arctic Ocean, covering , making it the world's second-largest country by total area and the fourth-largest country by land area. Canada's border with the United States is the world's longest land border. The majority of the country has a cold or severely cold winter climate, but southerly areas are warm in summer. Canada is sparsely populated, the majority of its land territory being dominated by forest and tundra and the Rocky Mountains. About four-fifths of the country's population of 36 million people is urbanized and live near the southern border. Its capital is Ottawa, its largest city is Toronto; other major urban areas include Montreal, Vancouver, Calgary, Edmonton, Quebec City, Winnipeg and Hamilton.")
        document = nlp(text)
        #print(document) # prints 'text'

        
        print("\n--- SENTENCE SPLITTING and AddToGraph: ---\n")
        for sentenceIndex, sentence in enumerate(document, start = 0):
            id2sentence[indexer(docNumber, sentenceIndex)] = str(sentence)
            graph.addNode(indexer(docNumber, sentenceIndex))
            if sentenceIndex != 0:
                graph.addEdge((indexer(docNumber, sentenceIndex-1), indexer(docNumber, sentenceIndex)))

        '''
        import stanfordnlp

        nlpU = stanfordnlp.Pipeline(processors='tokenize', lang='en')
        documentU = nlpU(text)
        frasi = list()
        for i, sentence in enumerate(documentU.sentences):
            print(f"====== Sentence {i+1} tokens =======")
            #print(*[f"index: {token.index.rjust(3)}    token: {token.text}" for token in sentence.tokens], sep='\n')
            for token in sentence.tokens:
                print(token.text)
            frasi.append(sentence)
        print("\n--- Frasi: ---\n")
        print(frasi)
        '''
        
        '''
        print("\n--- SENTENCE SPLITTING and AddToGraph: ---\n")
        for sentenceIndex, sentence in enumerate(tokenize.sent_tokenize(text), start=0):
            #print("\n---------------------\n")
            #print(sentence)
            #print("\n---------------------\n")
            id2sentence[indexer(docNumber, sentenceIndex)] = sentence
            graph.addNode(indexer(docNumber, sentenceIndex))
            if sentenceIndex != 0:
                graph.addEdge((indexer(docNumber, sentenceIndex-1), indexer(docNumber, sentenceIndex)))
        '''

        print("\n--- id2sentence: ---\n")
        print(id2sentence)
        

        print("\n--- Coreference resolution: ---\n")
        for var in document.coref_chains:
            print(var)



        print("\n--- Coreference resolution chain: ---\n")
        list_of_listOfCorefSentenceIndexes = list()
        for i, chain in enumerate(document.coref_chains, start=0):
            #print()
            #print(chain._coref.mention)
            list_of_listOfCorefSentenceIndexes.append(list())
            for mention in chain._coref.mention:
                list_of_listOfCorefSentenceIndexes[i].append(mention.sentenceIndex)


        print()
        print("Prima: ", list_of_listOfCorefSentenceIndexes)
        temp_list = list()
        for listElem in list_of_listOfCorefSentenceIndexes:
            temp_list.append(list(set(listElem)))  # eliminate dupicates and sort
            if len(temp_list[-1]) == 1:
                temp_list.pop(-1)

        list_of_listOfCorefSentenceIndexes = temp_list
        print("Risultato: ", list_of_listOfCorefSentenceIndexes)


        #for listElem in list_of_listOfCorefSentenceIndexes:

        for listElem in list_of_listOfCorefSentenceIndexes:
            for index, i in enumerate(listElem, start=0):
                if index == 0:
                    listElemPopped = listElem.copy()
                #print('\nPrima di pop',i,': ', listElemPopped)
                listElemPopped.pop(0)
                #print('\nDopo pop',i,': ', listElemPopped)
                for j in listElemPopped:
                    graph.addEdge((indexer(docNumber, i), indexer(docNumber, j)))
                    print('\nEdges: ', graph.getEdges())



        print("\n--- Graph print: ---\n")
        print('Nodes: ', graph.getNodes())
        print('Edges: ', graph.getEdges())

        radix_nodes = graph.getNodes()

        for (node1, node2) in graph.getEdges():
            if node2 in radix_nodes:
                radix_nodes.remove(node2)

        print("\n--- Radix nodes: ---\n")
        print(radix_nodes)

        
        print("\n--- Named entity recognition sentence level: ---\n")
        for i, sentence in enumerate(document, start=0):
            #first_sentence = document[0]
            print('\n ',i,') ')
            #id2entities[indexer(docNumber, i)] = str(sentence.entities)
            id2entities[indexer(docNumber, i)] = list()
            for entity in sentence.entities:
                id2entities[indexer(docNumber, i)].append(str(entity))
                print(entity, '({})'.format(entity.type))


        print('\n\n********* Ending document ', docNumber, ' *********\n\n')

    # ENITTY LINKING with the QUESTION
    questionNLP = nlp(question)
    questionEntities = list()
    for entity in questionNLP.entities:
        questionEntities.append(str(entity))
    questionLinked = False
    sentenceIDs = list(id2sentence.keys())
    sentenceIDs.remove('q')
    if(questionEntities != []):
        for j in sentenceIDs:
            if shareEntities(questionEntities, id2entities[j]):
                graph.addEdge(('q', getRadixNode(j)))
                questionLinked = True
    if not questionLinked:
        for j in sentenceIDs:
            if shareWords(question, id2sentence[j]):
                graph.addEdge(('q', getRadixNode(j)))

    # ENITTY LINKING between SENTENCES
    for i in sentenceIDs:
        for j in sentenceIDs:
            if i != j:
                if shareEntities(id2entities[i], id2entities[j]):
                    graph.addEdge((i, getRadixNode(j)))


    print("\n--- Id 2 entities: ---\n")
    print(id2entities)

    print("\n--- Graph: ---\n")
    print(graph.getEdges())


    file = open("CoreferenceGraph.pkl", "wb")
    pickle.dump(graph, file)
    file.close()
    file = open("id2sentence.pkl", "wb")
    pickle.dump(id2sentence, file)
    file.close()

    return graph, id2sentence, []

    

