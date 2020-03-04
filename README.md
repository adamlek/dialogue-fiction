# dialogue-fiction
Extracting Speakers and Addresses from Literary Fiction

annotated_text.tar.gz contain the annotated data for the project. 
annotated_text/AUTHOR/ include annotated chapters for each author.

# Annotation guidelines (Swedish)
1. Definitioner

YTTRANDE likställer vi här med mening.
Exempel:
– Ah, du pratar, svarade Sellén!

REPLIK är ett eller flera yttranden som börjar med tankstreck.
Exempel på två successiva repliker (och två successiva turer) med ett respektive två yttranden:
– Ah, du pratar, svarade Sellén!
– Gör som jag säger, din galning, annars får du inte sälja. Sätt dit en figur, en flicka, jag skall hjälpa dig om du inte kan, se här ...

TUR är en eller flera sammanhängande repliker.
Exempel på en tur som består av tre repliker:
– Han ser ut som en tjuv, sade Sellén, där han stod i fönstret och tittade illmarigt ut åt vägen. – Får han bara gå i fred för polisen nu, så är det bra! – Raska på Olle! ropade han efter den bortgående! Köp sex franska bröd och två halva öl, om det blir något över på färgen!

2. Annotering

Vi annoterar varje replik (dvs en tur kan ha flera annoteringar).
Vi skiljer mellan en eller flera adressater.

Tagg struktur:
    <TALARE--ADDRESSAT><TALARE_INDIKATOR--ADDRESSAT_INDIKATOR><KOMMENTAR:>

Talare--adressat (en adressat): <Sellén--Lundell>
Talare--adressat (mer än en adressat): <Sellén--FLERA>
Talare--adressat (oklar adressat eller oklara adressater): <Sellén--???>

Vi annoterar hur talaren och addressaten indikeras i texten (jämför He et al. 2013).
Implicit: <Sellén--Lundell><IMP--IMP> (Talaren/addressaten indikeras ej.)
Explicit: <Sellén--Lundell><EXP--EXP> (Talaren/addresaten indikeras med egennamn, för talare gäller denna endast när egennamnet förekommer med ett talverb.)
Anaforisk (anafor i samma stycke): <Sellén--Lundell><ANA--ANA> (e.g. – Ja, kalla in en av dem då, sade _han_)
Bestämd beskrivning: <Planertz--Cello><DESC--EXP> (e.g. – Adjö, herr Erlandsson, upprepade _kommunalmannen_)

Vi kan sätta in kommentarer till annoteringen: <Sellén--FLERA><IMP><Comment: först Sellén--Lundell, sedan Sellén--FLERA, sedan Sellén--Montanus>



3. Problemfall

a) Adressaten skiftar under en replik.

– Ah, det får han betalt för sen! Men, det här blir ingenting! Vi måste ta ett par lakan ur sängen! Vad gör det? Inte behöver vi några lakan! Se, så Olle! Stopp in bara! <Sellén--Lundell> <Comment: sedan Sellén--Montanus>

b) Adressaten skiftar under ett yttrande.

– Ja vad gör det. Han ska få betalt för dem, sen! Vad är det här för ett paket! En sammetsväst! Den var vacker! Den tar jag själv, så får Olle bära bort min! Kragar och manschetter! Ah, det är bara papper! Ett par strumpor! Se där Olle, tjugofem öre! Lägg dem i västen! Tombuteljerna får du lov att sälja! Jag tror det är så gott du säljer alltihop!

c) Addressaten skiftar mellan repliker inom en tur.

– Han ser ut som en tjuv, sade Sellén, där han stod i fönstret och tittade illmarigt ut åt vägen. <Sellén--Lundell><EXP--IMP> – Får han bara gå i fred för polisen nu, så är det bra! <Sellén--Lundell><IMP--IMP> – Raska på Olle! ropade han efter den bortgående! Köp sex franska bröd och två halva öl, om det blir något över på färgen! <Sellén--Olle><ANA--EXP>

d) I vissa repliker (i vissa verk) finns inte tankstreck utan citationstecken. 

Citationstecken byts ut mot tankstreck under preprocessing.

4. Andra uppgifter

a) Om – (tankstreck) finns inom en tur och inte indikerar en addressatskifte, byt ut – mot =.

Exempel:
– Är det så svårt? frågade Rissen och trummade mot bordskanten på ett sätt som irriterade mig. Är det verkligen så svårt att komma underfund med det? Tillåt mig en fråga – ja, ni behöver inte besvara den, om ni inte har lust – men anser ni mened vara av ondo under vilka omständigheter som helst? 

Skall transformeras till:

– Är det så svårt? frågade Rissen och trummade mot bordskanten på ett sätt som irriterade mig. Är det verkligen så svårt att komma underfund med det? Tillåt mig en fråga = ja, ni behöver inte besvara den, om ni inte har lust = men anser ni mened vara av ondo under vilka omständigheter som helst?
