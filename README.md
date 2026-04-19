# dataVisPy

Interaktive Python/Dash-Anwendung zur statistischen Auswertung von MEG-Konnektivitaetsdaten auf Basis von JSON- und CSV-Exporten.

## Was Die Software Macht

Die Anwendung dient zur interaktiven Auswertung von task-basierten MEG-Konnektivitaetsdaten, aktuell insbesondere fuer Coherence- und verwandte Konnektivitaets-Exporte aus JSON-Dateien in Kombination mit einer begleitenden `info.csv`.

Im Kern verbindet die Software drei Ebenen:

- Einlesen und Vereinheitlichen von Konnektivitaetsdaten aus JSON und Probanden-/Verhaltensdaten aus CSV
- Berechnung von subjektbezogenen Kontrastwerten zwischen zwei ausgewaehlten Trials fuer jedes ROI-Paar
- Statistische Auswertung dieser Kontraste auf Gruppen-, Longitudinal-, Kovariaten- und Netzwerkebene

Der typische Analyseablauf ist:

1. Auswahl eines JSON-Exports und einer zugehoerigen `info.csv`
2. Auswahl eines Frequenzbands, das ueber einzelne Frequenzbins gemittelt wird
3. Auswahl von zwei Trials, aus denen pro Person und ROI-Paar ein Delta `Trial B - Trial A` berechnet wird
4. Optional Einschraenkung auf bestimmte Gruppen oder Messzeitpunkte
5. Darstellung der Ergebnisse als Heatmap, Detailplot und Netzwerk-/Kovariatenanalyse

Die Heatmap zeigt nicht einfach rohe Konnektivitaetswerte, sondern inferenzfaehige Kontrastgroessen pro ROI-Paar. Je nach Modus werden Effektgroessen, p-Werte oder korrigierte q-Werte dargestellt. Signifikante Zellen koennen zusaetzlich anhand eines frei waehlbaren Schwellenwerts markiert werden.

Neben der Kantenebene besitzt die App jetzt auch einen eigenen `Network`-Tab. Dort werden aus den subjektweisen Delta-Matrizen kompaktere Netzwerkkennwerte berechnet, damit man nicht nur einzelne Kanten, sondern systemische Muster betrachten kann.

Zusaetzlich gibt es nun einen `Patterns`-Tab fuer explorative Mustererkennung, Embeddings, Clusterbildung und multivariate Brain-Behavior-Zusammenhaenge auf Basis der subjektweisen Edge-Delta-Profile.

## Statistische Moeglichkeiten

Die aktuell implementierten Statistikfunktionen sind auf ROI-Paar-Ebene organisiert. Fuer jedes ROI-Paar wird zunaechst ein subjektbezogener Kontrast berechnet; darauf aufbauend werden die folgenden Auswertungen angeboten.

### 1. Deskriptive Statistik pro ROI-Paar

Fuer jede analysierte Kante zwischen zwei ROIs berechnet die Software:

- `n`
- Mittelwert des Deltas
- Median
- Standardabweichung
- Standardfehler des Mittelwerts
- standardisierte Effektgroesse

Je nach Analysemodus werden diese Kennwerte fuer eine Gruppe, fuer zwei Gruppen oder fuer Kombinationen aus Gruppe und Zeitpunkten berichtet.

### 2. Inferenzstatistik fuer Trial-Kontraste

Im Standardmodus wird pro Person und ROI-Paar ein Trial-Kontrast `Trial B - Trial A` gebildet. Darauf stehen derzeit folgende Tests zur Verfuegung:

- Ein-Gruppen-Fall: gepaarter t-Test
  Verwendet, wenn innerhalb einer einzelnen Gruppe oder ueber alle Gruppen hinweg geprueft wird, ob der mittlere Trial-Kontrast von null abweicht.
- Zwei-Gruppen-Fall: Welch-t-Test
  Verwendet, wenn die Trial-Kontraste zweier Gruppen miteinander verglichen werden. Der Test ist robust gegen ungleiche Varianzen und unterschiedliche Gruppengroessen.

### 3. Longitudinale Statistik

Die Software unterstuetzt auch longitudinale Delta-vs-Delta-Analysen. Dabei wird zunaechst innerhalb jedes Messzeitpunkts der Trial-Kontrast `Trial B - Trial A` berechnet. Anschliessend werden diese Trial-Deltas ueber zwei ausgewaehlte Zeitpunkte oder andere longitudinale Zustandswerte verglichen.

Es gibt zwei Modi:

- Gepaarter longitudinaler Modus
  Es werden nur Personen beruecksichtigt, die an beiden ausgewaehlten Zeitpunkten vorliegen. Anschliessend wird die Differenz der Trial-Deltas innerhalb derselben Person berechnet und mit einem gepaarten t-Test ausgewertet. Wenn zwei Gruppen gewaehlt sind, wird auf diesen longitudinalen Differenzen ein Welch-t-Test zwischen den Gruppen gerechnet.
- Ungepaarter longitudinaler Modus
  Es werden alle verfuegbaren Messungen an den beiden gewaehlten Zeitpunkten verwendet, auch wenn nicht jede Person zweimal gemessen wurde. Bei einer Gruppe erfolgt der Vergleich der Zeitpunkte mit einem Welch-t-Test auf den Trial-Deltas. Bei zwei Gruppen wird ein lineares OLS-Modell mit `group`, `timepoint` und `group x timepoint` verwendet; der zentrale Test ist dann der Interaktionseffekt.

Damit deckt die App sowohl klassische wiederholte Messungen als auch unvollstaendige longitudinale Stichproben ab.

Zusaetzlich besitzt der Heatmap-Bereich jetzt einen eigenen Block `Longitudinal modeling` fuer die aktuell ausgewaehlte Kante. Dort werden staerkere, edge-spezifische Langzeitmodelle gerechnet, waehrend die eigentliche Heatmap weiterhin den schnellen Ueberblick auf ROI-Paar-Ebene liefert.

Im Longitudinal-Block sind aktuell verfuegbar:

- `Mixed effects`
  Ein Mixed-Effects-Modell fuer die ausgewaehlte Kante mit Random Intercept pro `IDX`, festen Effekten fuer Zeitpunkt, optional Gruppe und Gruppe-x-Zeit-Interaktion sowie optionalen numerischen Kovariaten. Ein Random Slope fuer Zeit kann angefordert werden; wenn die Datenlage dafuer nicht ausreicht, faellt das Modell kontrolliert auf Random-Intercept-only zurueck.
- `Change score`
  Robustheitsanalyse auf Basis der subjektweisen Veraenderung `Follow-up - Baseline` des ausgewaehlten Edge-Deltas.
- `ANCOVA`
  Baseline-adjustierte Follow-up-Analyse fuer dieselbe ausgewaehlte Kante.
- `Longitudinal trajectories`
  Individuelle Linien pro Person, Gruppentrend und Konfidenzband fuer die ausgewaehlte Kante. Wenn mehr als zwei Zeitpunkte vorhanden sind, werden diese als Kontext mit dargestellt.
- `Reliable change`
  Approximate Reliable-Change-Analyse fuer die ausgewaehlte Kante auf Basis gepaarter Subjekte. Die aktuelle Version verwendet eine Jacobson-Truax-aehnliche Schaetzung mit Baseline-SD und einem explizit berichteten angenommenen Reliabilitaetskoeffizienten.

### 4. Korrektur fuer multiples Testen

Da pro Analyse viele ROI-Paare gleichzeitig getestet werden, koennen p-Werte direkt oder multipel korrigiert betrachtet werden. Aktuell sind implementiert:

- `None`
- `FDR-BH`
- `Bonferroni`
- `Holm`

Die Korrektur erfolgt ueber alle getesteten ROI-Paare der aktuellen Analyse. In der Heatmap koennen sowohl rohe p-Werte als auch korrigierte q-Werte dargestellt werden.

### 5. Korrelationsanalysen mit Verhaltens- und Kovariatenwerten

Die bisherigen Kovariatenanalysen bleiben erhalten, sind aber nicht mehr als eigener Haupt-Tab organisiert. Stattdessen liegen sie direkt in den Heatmap-Detailbereichen fuer die ausgewaehlte Kante.

Fuer ein ausgewaehltes ROI-Paar kann der berechnete Delta-Wert mit numerischen Variablen aus der `info.csv` in Beziehung gesetzt werden. Verfuegbar sind:

- Pearson-Korrelation
- Spearman-Korrelation

Zusaetzlich berechnet die Software fuer die ausgewaehlte Beziehung:

- Regressionsgerade
- Steigung und Achsenabschnitt
- p-Wert der linearen Anpassung
- `R^2`
- RMSE

Wenn zwei Gruppen vorhanden sind, werden die Korrelationen auch gruppenweise berechnet. Liegen genau zwei Gruppen mit ausreichender Fallzahl vor, vergleicht die App die Korrelationskoeffizienten zusaetzlich mit einem Fisher-z-Test. Bei Spearman wird dieser Vergleich als Approximation berichtet.

### 6. Regressionsmodelle fuer ein ausgewaehltes ROI-Paar

Es gibt zwei Regressions-Ebenen:

- Einfache Kovariaten-Regression auf den ausgewaehlten Edge-Delta-Wert
  Hier wird ein OLS-Modell gerechnet, bei dem das Edge-Delta die Zielvariable ist und frei waehlbare numerische Kovariaten als Praediktoren verwendet werden.
- Design-bewusste multivariate Regression
  Dieses Modell beruecksichtigt automatisch, in welchem Analysemodus man sich befindet. Je nach Setting enthaelt das Modell zusaetzlich explizite Designterme fuer Gruppe, Zeitpunkt oder Gruppe-x-Zeitpunkt-Interaktion.

Die multivariate Regression kann je nach Setting folgende Zielvariablen modellieren:

- den ausgewaehlten Edge-Delta-Wert selbst
- eine numerische Variable aus der `info.csv`

Als Praediktoren koennen verwendet werden:

- numerische Kovariaten aus der `info.csv`
- optional der Edge-Delta-Wert selbst, wenn eine andere Zielvariable modelliert wird
- je nach Analysemodus automatische Designterme fuer Gruppe, Zeitpunkt oder Interaktion

Berichtet werden:

- Regressionskoeffizienten
- Standardfehler
- t-Werte
- p-Werte
- Konfidenzintervalle
- `R^2`
- adjustiertes `R^2`
- beobachtete vs. vorhergesagte Werte
- Residuenplots

Gerade im longitudinalen Modus ist das hilfreich, weil man nicht nur unadjustierte Gruppen- oder Zeitunterschiede sehen kann, sondern auch alters-, verhaltens- oder klinisch adjustierte Effekte.

Im neuen Longitudinal-Block der Heatmap werden diese Regressionsideen fuer die aktuell selektierte Kante um Mixed-Effects-, Change-score- und ANCOVA-Modelle erweitert. Dadurch kann man longitudinale Haupteffekte und Robustheit direkt an der interessierenden Kante vergleichen, ohne die gesamte Heatmap auf schwerere Modelle umzustellen.

### 7. Netzwerk-Panel

Der `Network`-Tab verschiebt die Perspektive weg von einzelnen Kanten hin zu ROI- und Netzwerkmustern. Grundlage ist immer die subjektbezogene Delta-Matrix `Trial B - Trial A` der aktuellen Auswahl.

Oben im Panel befindet sich zusaetzlich eine aufklappbare `Help / Explanation`-Sektion. Dort werden die Modi, Kennwerte, Schaetzungen, Schwellwerte und Abkuerzungen der Netzwerk-Analyse direkt in der App erklaert.

Aktuell verfuegbar sind:

- ROI-Summary-Scores wie mittlere Konnektivitaet eines ROI zu allen anderen
- Netzwerk-Summary-Scores wie within-network und between-network connectivity fuer anatomisch aus ROI-Namen abgeleitete Subnetze
- Lateralisierungsindizes auf globaler und homologer ROI-Ebene
- Graphmetriken wie node strength, degree, clustering coefficient, local/global efficiency, betweenness, participation und hubness
- explorative Modul-/Community-Ansichten auf Basis supraschwelliger Komponenten
- eine erste `NBS`-Pipeline mit primaerer Testschwelle, Zusammenhangskomponenten und permutationsbasierter Komponentensignifikanz

Methodische Hinweise:

- Funktionelle Netzwerke wie Default-Mode oder fronto-parietale Systeme werden in dieser Version noch nicht explizit modelliert. Die Netzwerklabels sind derzeit rein anatomisch aus ROI-Namen abgeleitet.
- Fuer Graphmetriken wird standardmaessig ein gewichteter Ansatz verwendet. Negative Delta-Werte werden nicht ungefiltert in alle Metriken eingespeist; je nach Einstellung werden positive oder absolute Gewichte verwendet.
- Schwellen koennen optional ueber absolutes Gewicht oder Dichte gesetzt werden. Community-Analyse und NBS sind deshalb bewusst separat von den normalen ROI-Paar-q-Werten organisiert.
- NBS ist inferenziell auf Netzwerkebene gedacht und ersetzt nicht die klassische Edge-Heatmap.

#### Modi und Werkzeuge im Network-Panel

Der Tab besitzt derzeit vier Modi:

- `Summary`
  Fokus auf kompakte ROI-, Netzwerk- und Lateralisierungs-Scores mit Inferenzstatistik.
- `Graph`
  Fokus auf node-wise und globale Graphmetriken.
- `Modules`
  Explorative Darstellung von Komponenten bzw. Modulen auf Basis der aktuell geschwellten Netzwerkstruktur.
- `NBS`
  Netzwerkbasierte Inferenz auf Ebene zusammenhaengender supraschwelliger Kantenkomponenten.

Die zentralen Eingabefelder im Panel sind:

- `Score family / metric`
  Waehlt aus, welche Score-Familie oder Graphmetrik in Tabellen und Diagrammen hervorgehoben wird.
- `Weight mode`
  Definiert, wie Delta-Werte als Gewichte in die Graphanalyse eingehen.
- `Threshold mode`
  Legt fest, ob keine Schwelle, eine absolute Gewichtsschwelle oder eine Dichteschwelle verwendet wird.
- `Threshold value`
  Parameter fuer die aktuelle Schwellenregel.
- `NBS primary threshold`
  Teststatistik-Schwelle fuer supraschwellige Kanten in der NBS-Analyse.
- `NBS permutations`
  Anzahl der Permutationen fuer die komponentenbasierte Signifikanzschaetzung.

#### Erklerung der Netzwerkmasse

Summary-Masse:

- `ROI mean connectivity`
  Mittleres Delta eines ROI zu allen anderen ROIs.
- `Ipsilateral mean`
  Mittleres Delta eines ROI zu ROIs derselben Hemisphaere.
- `Contralateral mean`
  Mittleres Delta eines ROI zu ROIs der gegenueberliegenden Hemisphaere.
- `Within-class mean`
  Mittleres Delta innerhalb derselben anatomischen ROI-Klasse.
- `Between-class mean`
  Mittleres Delta zwischen einer ROI-Klasse und anderen anatomischen Klassen.
- `Within-network connectivity`
  Mittleres Delta innerhalb eines anatomisch definierten Subnetzes.
- `Between-network connectivity`
  Mittleres Delta zwischen zwei anatomischen Subnetzen.
- `Laterality index`
  Normierter Links-Rechts-Unterschied, typischerweise `(left - right) / (|left| + |right|)`.

Graphmetriken:

- `Node strength`
  Summe aller an einem ROI anliegenden Kantengewichte.
- `Degree`
  Anzahl der nach Schwellenung verbleibenden Kanten eines ROI.
- `Clustering coefficient`
  Ausmass, in dem Nachbarn eines ROI ebenfalls untereinander verbunden sind.
- `Local efficiency`
  Effizienz der Kommunikation innerhalb der Nachbarschaft eines ROI.
- `Global efficiency`
  Integrationsmass des Gesamtnetzwerks auf Basis inverser kuerzester Wege.
- `Betweenness`
  Anteil kuerzester Wege, auf denen ein ROI als Vermittler liegt.
- `Participation coefficient`
  Wie breit ein ROI ueber mehrere Komponenten bzw. Module hinweg verbindet.
- `Hubness`
  Zusammengesetztes Zentralitaetsmass auf Basis mehrerer node-wise Kennwerte.
- `Modularity`
  Ausmass der Trennbarkeit von Modulen oder Komponenten in der gewichteten Netzwerkstruktur.

#### Weighting, Thresholding und NBS

- `Positive weights`
  Negative Delta-Werte werden fuer Graphmetriken auf null gesetzt.
- `Absolute weights`
  Es wird die absolute Delta-Groesse als Gewicht verwendet.
- `Raw positive part`
  Praktisch eine positive, gerichtete Gewichtsauslegung fuer die derzeitige Graphanalyse.
- `None`
  Keine zusaetzliche Schwellenung.
- `Absolute weight`
  Es bleiben nur Kanten mit mindestens dem gewaehlten Gewicht erhalten.
- `Density`
  Es bleibt nur der staerkste Anteil aller Kanten erhalten.

Bei `NBS` gilt:

- Einzelne Kanten werden zuerst mit einer primaeren Teststatistik-Schwelle gefiltert.
- Benachbarte supraschwellige Kanten werden zu Komponenten zusammengefasst.
- Die Signifikanz wird permutationsbasiert auf Komponentenebene berechnet.
- Die resultierenden p-Werte gelten fuer Netzwerk-Komponenten, nicht fuer einzelne Kanten.

#### Abkuerzungen im Network-Panel

- `ROI`
  Region of interest
- `Delta`
  Subjektbezogener Kontrast `Trial B - Trial A`
- `LI`
  Laterality index
- `NBS`
  Network-Based Statistic
- `n`
  Anzahl verfuegbarer Beobachtungen fuer den jeweiligen Score
- `p`
  Roher p-Wert
- `q`
  Multipel korrigierter p-Wert

### 8. Outlier-Panel

Der `Outlier`-Tab dient dazu, den Einfluss einzelner Personen (`IDX`) auf die aktuellen Ergebnisse systematisch zu untersuchen. Er arbeitet immer auf Basis der gerade gewaehlten Analysekonfiguration aus Sidebar und Heatmap.

Das Panel besitzt drei Modi:

- `Global influence`
  Untersucht, wie stark sich die Gesamtanalyse aendert, wenn jeweils eine Person ausgeschlossen wird.
- `Selected ROI pair`
  Zeigt den Einfluss einzelner Personen speziell auf die aktuell in der Heatmap ausgewaehlte Kante.
- `Regression diagnostics`
  Zeigt Diagnosemasse fuer das Regressionsmodell der aktuell ausgewaehlten Kante.

Werkzeuge im Outlier-Panel:

- `Exclude persons globally (IDX)`
  Schaltet einzelne Personen fuer die komplette laufende Analyse aus.
- `Reset exclusions`
  Entfernt alle aktuell gesetzten globalen Ausschluesse.
- `Mark top 3 influential`
  Markiert automatisch die drei Personen mit dem groessten globalen Einfluss nach den aktuellen Influence-Kriterien.

#### Inhalte im Modus `Global influence`

Hier wird fuer jede verbleibende Person eine Leave-one-out-Analyse gerechnet. Berichtet werden unter anderem:

- `Sig. loss`
  Anzahl signifikanter Zellen, die nach Ausschluss dieser Person nicht mehr signifikant sind.
- `Sig. gain`
  Anzahl Zellen, die erst nach Ausschluss dieser Person signifikant werden.
- `Sig. switches`
  Gesamtzahl von Signifikanzwechseln.
- `Mean |effect size change|`
  Mittlere absolute Aenderung der Effektgroessen ueber alle ROI-Paare.

Die zugehoerige Grafik fasst zusammen, welche Personen die globale Netzwerk- oder Edge-Inferenz am staerksten beeinflussen.

#### Inhalte im Modus `Selected ROI pair`

Dieser Modus fokussiert auf das aktuell in der Heatmap gewaehlte ROI-Paar. Fuer jede Person wird geprueft, wie sich beim Ausschluss dieser Person die Kennwerte dieser einen Kante aendern.

Berichtet werden zum Beispiel:

- Aenderung des mittleren Deltas
- Aenderung der Effektgroesse
- Aenderung von `p` und `q`
- Ob die Signifikanz fuer genau diese Kante umschlaegt
- Falls aktiv: Aenderung des primaeren Regressionskoeffizienten der ausgewaehlten Heatmap-Regression

Das ist hilfreich, um zu erkennen, ob ein auffaelliges Heatmap-Ergebnis auf sehr wenige Personen zurueckgeht.

#### Inhalte im Modus `Regression diagnostics`

Dieser Modus bezieht sich auf die multivariate Regression im Heatmap-Bereich fuer die aktuell ausgewaehlte Kante. Gezeigt werden:

- `Leverage`
  Wie stark ein Punkt aufgrund seiner Praediktorlage das Modell beeinflussen kann.
- `Studentized residual`
  Standardisierter Residuenwert zur Erkennung ausreissender Beobachtungen.
- `Cook's distance`
  Kombiniertes Einflussmass, das Residuum und Leverage verbindet.

Die Tabellen zeigen die auffaelligsten Beobachtungen inklusive `IDX`, Residuum, Leverage und `Cook's D`.

#### Abkuerzungen im Outlier-Panel

- `IDX`
  Stabile Personen-ID ueber Messzeitpunkte hinweg
- `Sig.`
  Signifikanz
- `q`
  Multipel korrigierter p-Wert
- `Cook's D`
  Cook's distance

### 9. Patterns-Panel

Der `Patterns`-Tab dient dazu, Konnektivitaetsmuster nicht nur zu testen, sondern als Struktur im gesamten Datensatz sichtbar zu machen. Grundlage ist wieder die aktuelle Analyseauswahl, aber statt einzelne Kanten oder wenige Netzwerkkennwerte anzusehen, wird pro Beobachtung der gesamte Vektor aller Edge-Deltas verwendet.

Je nach Analysemodus ist eine Beobachtung:

- eine Person im Standardmodus
- eine Person mit gepaartem longitudinalem Delta-vs-Delta
- ein Subjekt-Zeitpunkt im ungepaarten longitudinalen Modus

Oben im Tab befindet sich ebenfalls eine aufklappbare `Help / Explanation`-Sektion.

#### Modi im Patterns-Panel

- `Embedding`
  Niedrigdimensionale Projektion der Subjekte auf Basis aller Edge-Deltas.
- `Subject clusters`
  Gruppierung von Personen mit aehnlichen Konnektivitaetsprofilen.
- `Feature patterns`
  Clustering von Kanten oder ROI-Profilen, die sich ueber Personen hinweg gemeinsam veraendern.
- `CCA / PLS`
  Multivariate Beziehung zwischen Konnektivitaetsmustern und numerischen Verhaltens- oder Klinikscores.

#### Werkzeuge im Patterns-Panel

- `Embedding method`
  Waehlt die Projektionsmethode. In der aktuellen dependency-armen Version ist `PCA` voll verfuegbar; `t-SNE` und `UMAP` geben derzeit einen Hinweis zur fehlenden Verfuegbarkeit.
- `Embedding dimension`
  2D oder 3D Darstellung der Projektion.
- `Color variable`
  Farbgebung nach Gruppe, Zeit, `IDX` oder numerischer Kovariate.
- `Cluster method`
  Aktuell unterstuetzt: `k-means` und `hierarchical`. `Gaussian mixture` ist als Platzhalter sichtbar, benoetigt aber zusaetzliche Abhaengigkeiten.
- `Number of clusters`
  Anzahl der Zielcluster fuer die Subjekt-Clusterung.
- `Feature pattern level`
  Definiert, ob Kanten oder ROI-Profile gemeinsam geclustert werden.
- `Multivariate method`
  Aktuell ist `PLS` direkt nutzbar. `CCA` ist in dieser Version als Option sichtbar, aber noch nicht aktiv ohne weitere Stabilisierung bzw. zusaetzliche Tools.
- `Behavior variables`
  Auswahl numerischer Variablen aus der `info.csv` fuer den multivariaten Brain-Behavior-Block.
- `Components`
  Anzahl der latenten Komponenten fuer PCA oder PLS-nahe Auswertungen.
- `Standardization`
  Aktuell `none` oder `z-score by feature`.

#### Inhalte im Modus `Embedding`

Hier wird jeder Beobachtungspunkt anhand seines kompletten Edge-Delta-Vektors in einen niedrigdimensionalen Raum projiziert.

Aktuell verfuegbar:

- `PCA`
  Robuste lineare Hauptkomponentenanalyse ohne zusaetzliche externe Libraries.

Berichtet und visualisiert werden:

- 2D- oder 3D-Scatterplot der Beobachtungen
- Farbkodierung nach Gruppe, Zeit oder numerischer Variable
- erklaerte Varianz der ersten Komponenten
- staerkste Kanten-Loadings pro Komponente

Das ist hilfreich, um Cluster, Gradienten, Outlier und Gruppen-Trennung im Gesamtraum zu sehen.

#### Inhalte im Modus `Subject clusters`

Hier werden Personen bzw. Beobachtungen anhand ihrer vollstaendigen Konnektivitaetsprofile gruppiert.

Aktuell verfuegbar:

- `k-means`
- `hierarchical` Clustering

Berichtet werden:

- Clusterzuordnung jeder Beobachtung
- Visualisierung der Cluster im Embedding-Raum
- Clusteruebersicht mit Cluster-Groesse, Gruppenverteilung und vorhandenen Zeitpunkten

Das ist besonders nuetzlich, um Subgruppen wie moegliche Responder/Non-responder oder unterschiedliche Netzwerk-Phaenotypen explorativ sichtbar zu machen.

#### Inhalte im Modus `Feature patterns`

Hier wird nicht ueber Personen, sondern ueber Features geclustert:

- auf `edges`-Ebene: welche Kanten veraendern sich gemeinsam?
- auf `ROIs`-Ebene: welche ROIs zeigen aehnliche Veraenderungsprofile?

Berichtet werden:

- geordnete Korrelationsmatrix
- hierarchische Sortierung der Features
- visuelle Hervorhebung gemeinsam variierender Feature-Bloecke

Dieser Modus hilft, funktionell zusammenhaengende Veraenderungsmuster im Datenraum zu erkennen, auch wenn einzelne Kanten fuer sich isoliert wenig auffaellig sind.

#### Inhalte im Modus `CCA / PLS`

Dieser Block verknuepft hochdimensionale Konnektivitaetsmuster mit mehreren numerischen Variablen aus der `info.csv`.

Aktuell verfuegbar:

- `PLS`
  Als explorative latente Brain-Behavior-Zerlegung.

Berichtet werden:

- Brain-Scores und Behavior-Scores der ersten latenten Komponente
- Scatterplot `brain score` vs `behavior score`
- staerkste Konnektivitaetsgewichte
- staerkste Gewichte der Behavior-Variablen
- latente Singularwerte als Groessenordnung des gemeinsamen Musters

Dieser Modus ist nicht dasselbe wie einzelne Korrelationen einer Kante mit einem Score, sondern beschreibt einen multivariaten Zusammenhang zwischen vielen Kanten und mehreren Verhaltensvariablen gleichzeitig.

#### Methodische Hinweise zum Patterns-Panel

- Der gesamte Tab ist explorativ zu verstehen.
- Ergebnisse ersetzen keine inferenzielle Heatmap- oder Netzwerkstatistik.
- Fehlende Werte werden in der aktuellen Version ueber `complete cases` behandelt.
- `PCA`, `k-means`, hierarchisches Clustering und ein PLS-aehnlicher multivariater Block sind direkt verfuegbar.
- `t-SNE`, `UMAP`, `Gaussian mixtures` und direktes `CCA` sind im Interface vorbereitet, aber in dieser Version aufgrund fehlender Zusatzabhaengigkeiten noch nicht voll aktiviert.

#### Abkuerzungen im Patterns-Panel

- `PCA`
  Principal component analysis
- `PLS`
  Partial least squares
- `CCA`
  Canonical correlation analysis
- `ROI`
  Region of interest
- `Delta`
  Subjektbezogener Kontrast `Trial B - Trial A`

### 10. Was Die Statistik Aktuell Nicht Abdeckt

Die derzeitige Implementierung ist weiterhin klar parametrisch ausgerichtet. Der Schwerpunkt liegt auf t-Tests, linearen Korrelationen, OLS-Regressionsmodellen und ersten netzwerkbasierten Zusammenfassungen. Aktuell sind zum Beispiel noch nicht enthalten:

- gemischte lineare Modelle
- nichtparametrische Gruppenvergleiche wie Mann-Whitney oder Wilcoxon als Alternative zu den t-Tests
- explizite funktionelle Netzwerk-Mappings wie DMN oder fronto-parietale Kontrollnetzwerke
- vollstaendige, optimierte Community-Detection-Verfahren wie Louvain oder Leiden

Fuer explorative und interaktive Analyse ist der aktuelle Umfang bereits sehr nuetzlich; fuer spaetere methodische Erweiterungen ist aber noch Luft nach oben.

## Lokal starten

```bash
source .venv/bin/activate
python app.py
```

Optional mit `src`-Layout direkt als Modul:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m data_vis_py.app
```

## Tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

## Erwartete Datendateien

Fuer einen Datensatz wie `data/raw/REST_24_Stroke/` erwartet die App aktuell mindestens:

- `data_coh.json` oder `export_coh.json` mit den Konnektivitaetswerten
- `info.csv` mit den Probanden- und Verhaltensdaten

## Pflichtspalten in `info.csv`

Die ersten vier Spalten muessen immer vorhanden sein:

1. `ID`
   Diese Spalte enthaelt die eindeutige Messungs-ID, also genau die Kennung, die auch im JSON unter `subject_id` vorkommt.
2. `Group`
   Diese Spalte beschreibt die Gruppenzugehoerigkeit der Messung, zum Beispiel `Stroke`, `Control` oder numerische Gruppen wie `1` und `2`.
3. `IDX`
   Diese Spalte ist die stabile Personen-ID ueber mehrere Messungen hinweg. Gleicher `IDX` bedeutet gleiche Person. Diese Spalte wird fuer longitudinale Vergleiche verwendet und ist deshalb Pflicht.
4. `MTime`
   Diese Spalte beschreibt den Messzeitpunkt, zum Beispiel `1` und `2` oder `M1` und `M2`. Intern wird sie auf Werte wie `M1`, `M2` normalisiert.

## Weitere Spalten

Alle weiteren Spalten in `info.csv` werden als zusaetzliche Behavior- oder Kovariatenfelder behandelt. Sie koennen in der App verwendet werden fuer:

- Korrelationsanalysen
- lineare Regressionsmodelle
- longitudinale Selektion ueber eine frei waehlbare Spalte

Beispiele sind `age`, `MoCA`, `NIHSS`, `DSS`, `gender` oder projektspezifische Labordaten.

## Longitudinale Analyse

Wenn in der App die longitudinale Analyse aktiviert wird:

- waehlt man zuerst eine Spalte aus der Behavior-Tabelle
- danach zwei konkrete Werte aus dieser Spalte
- die App berechnet dann pro Person zunaechst den Trial-Kontrast `Trial B - Trial A`
- anschliessend wird dieses Delta zwischen den beiden ausgewaehlten Messungen derselben Person verglichen

Die Zuordnung derselben Person erfolgt dabei ausschliesslich ueber `IDX`.
