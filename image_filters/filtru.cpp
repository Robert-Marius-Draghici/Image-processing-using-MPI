#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#define TAG_HELPER 1
#define TAG_SOBEL 2
#define TAG_MEAN_REMOVAL 3
#define TAG_ACTIVE 4
#define TAG_TERMINATE 5
#define TAG_NUMBER 6
#define TAG_VECTOR 7

typedef struct {
	std::string header;
	int height;
	int width;
	int maxVal;
	int **pixelMatrix;
	std::vector<char> whitespaces;
	std::vector<int> pixelVector;
} Image;

Image readImageFromFile(std::string inputImageName);

void writeImageToFile(std::string outputImageName, Image image);

void sendDataToChildren(Image image, int helperInfo[3], int parent, int filterTag, bool isRoot, int *receivedPixels,
                        std::vector<int> neighbours);

void receiveDataFromChildren(Image &image, int helperInfo[3], int parent, int filterTag, bool isRoot,
                             std::vector<int> neighbours);

void receiveStatisticsFromChildren(std::vector<int> &ranks, std::vector<int> &statistics,
                                   std::vector<int> neighbours, int rank, int parent);

int main(int argc, char *argv[]) {
	int rank;
	int nProcesses;
	int **pixelMatrix;
	int height;
	int width;
	int *newPixels;
	int *receivedPixels;
	int filterTag;
	int value = 0;
	int maxVal;
	MPI_Init(&argc, &argv);
	MPI_Status status;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);
	
	std::string line;
	std::ifstream topologyFile(argv[1]);
	
	if (!topologyFile) {
		printf("Eroare la deschiderea fisierului %s!\n", argv[1]);
		exit(-1);
	}
	
	/*
	 * Fiecare proces citeste si ignora liniile din fisierul topologie.in, pana
	 * ajunge la linia cu ID-ul egal cu rankul sau.
	 */
	std::string lineNumber;
	do {
		std::getline(topologyFile, line);
		lineNumber = line.substr(0, line.find(":"));
	} while (atoi(lineNumber.c_str()) != rank);
	
	int position = line.find(":");
	line.erase(0, position + 1);
	
	std::vector<int> neighbours;
	std::stringstream ss(line);
	int number;
	while (!ss.eof()) {
		ss >> number;
		neighbours.push_back(number);
	}
	
	line.clear();
	
	/*
	 * Aici incepe blocul de cod specific radacinii.
	 */
	if (rank == 0) {
		std::ifstream imagesFile(argv[2]);
		
		if (!imagesFile) {
			printf("Eroare la deschiderea fisierului %s!\n", argv[2]);
			exit(-1);
		}
		
		std::getline(imagesFile, line);
		int nrInputImages = atoi(line.c_str());
		
		for (int i = 0; i < nrInputImages; i++) {
			std::getline(imagesFile, line);
			std::stringstream stream(line);
			std::string filter;
			std::string inputImageName;
			std::string outputImageName;
			stream >> filter >> inputImageName >> outputImageName;
			
			if (!strcmp(filter.c_str(), "sobel"))
				filterTag = TAG_SOBEL;
			else
				filterTag = TAG_MEAN_REMOVAL;
			
			Image image = readImageFromFile(inputImageName.c_str());
			int helperInfo[3];
			helperInfo[0] = image.height;
			helperInfo[1] = image.width;
			helperInfo[2] = image.maxVal;
			sendDataToChildren(image, helperInfo, -1, filterTag, true, NULL, neighbours);
			receiveDataFromChildren(image, helperInfo, -1, filterTag, true, neighbours);
			writeImageToFile(outputImageName, image);
			
			/*
			 * Este necesar sa eliberez memoria alocata, deoarece altfel
			 * se umple memoria cu foarte multe obiecte pana ajunge la overflow.
			 */
			for (int j = 0; j < image.height; j++) {
				free(image.pixelMatrix[j]);
				image.pixelMatrix[j] = NULL;
			}
			free(image.pixelMatrix);
			image.pixelMatrix = NULL;
			image.pixelVector.clear();
		}
		
		/*
		 * Dupa ce am terminat de prelucrat toate pozele, trimit tagul de terminare
		 * pentru a anunta procesele ca nu mai sunt poze de prelucrat si pot
		 * trimite statisticile.
		 */
		for (int i = 0; i < neighbours.size(); i++) {
			MPI_Send(&value, 1, MPI_INT, neighbours[i], TAG_TERMINATE, MPI_COMM_WORLD);
		}
		
		std::vector<int> ranks;
		std::vector<int> statistics;
		receiveStatisticsFromChildren(ranks, statistics, neighbours, rank, -1);
		
		/*
		 * Vectorul ranks cuprinde rankurile proceselor din topologie necesare
		 * pentru a eticheta statisticile, insa ordinea elementelor din vector
		 * este aleatoare. Folosesc un map pentru a avea rankurile in ordine
		 * crescatoare.
		 */
		std::map<int, int> s;
		
		for (int i = 0; i < nProcesses; i++)
			s[ranks[i]] = statistics[i];
		
		std::ofstream statisticsFile(argv[3]);
		for (std::map<int, int>::iterator it = s.begin(); it != s.end(); ++it)
			statisticsFile << it->first << ": " << it->second << "\n";
		
		statisticsFile.close();
		imagesFile.close();
		/*
		 * Aici se termina blocul de cod specific radacinii.
		 */
	} else {
		int tag;
		int parent;
		bool parentSet = false;
		/*
		 * Aici incepe blocul de cod specific unui nod intermediar.
		 */
		if (neighbours.size() > 1) {
			while (tag != TAG_TERMINATE) {
				/*
				 * Parintele unui nod trebuie setat doar o data, atunci cand
				 * primeste primul mesaj (parintele unui nod este considerat
				 * nodul care a trimis primul mesaj).
				 */
				if (!parentSet) {
					MPI_Recv(&value, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					parent = status.MPI_SOURCE;
					parentSet = true;
				} else {
					MPI_Recv(&value, 1, MPI_INT, parent, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				}
				tag = status.MPI_TAG;
				switch (tag) {
					case TAG_ACTIVE: {
						int helperInfo[3];
						MPI_Recv(&(helperInfo[0]), 3, MPI_INT, parent, TAG_HELPER, MPI_COMM_WORLD, &status);
						height = helperInfo[0];
						width = helperInfo[1];
						maxVal = helperInfo[2];
						receivedPixels = (int *) calloc(height * width, sizeof(int));
						MPI_Recv(&(receivedPixels[0]), height * width, MPI_INT, parent, MPI_ANY_TAG, MPI_COMM_WORLD,
						         &status);
						
						filterTag = status.MPI_TAG;
						/* 
						 * Nodurile intermediare nu au acces la poza, insa in functie
						 * trebuie sa trimitem o structura de tip Image, asa ca trimit
						 * o structura cu campurile vide.
						 */
						Image image = {};
						sendDataToChildren(image, helperInfo, parent, filterTag, false, receivedPixels, neighbours);
						receiveDataFromChildren(image, helperInfo, parent, filterTag, false, neighbours);
					}
						break;
					
					case TAG_TERMINATE: {
						for (int i = 0; i < neighbours.size(); i++)
							if (neighbours[i] != parent) {
								MPI_Send(&value, 1, MPI_INT, neighbours[i], TAG_TERMINATE, MPI_COMM_WORLD);
							}
						
						
						std::vector<int> ranks;
						std::vector<int> statistics;
						receiveStatisticsFromChildren(ranks, statistics, neighbours, rank, parent);
						
						/*
						 * Trimite statistica la parinte.
						 */
						int nr = ranks.size();
						MPI_Send(&nr, 1, MPI_INT, parent, TAG_VECTOR, MPI_COMM_WORLD);
						int *auxiliaryVector1 = ranks.data();
						MPI_Send(&(auxiliaryVector1[0]), nr, MPI_INT, parent, TAG_VECTOR, MPI_COMM_WORLD);
						int *auxiliaryVector2 = statistics.data();
						MPI_Send(&(auxiliaryVector2[0]), nr, MPI_INT, parent, TAG_VECTOR, MPI_COMM_WORLD);
					}
						break;
				}
			}
		/*
		 * Aici se termina blocul de cod specific unui nod intermediar.
		 */
		/*
		 * Aici incepe blocul de cod specific unui nod frunza.
		 */
		} else {
			int nrLinesProcessed = 0;
			while (tag != TAG_TERMINATE) {
				if (!parentSet) {
					MPI_Recv(&value, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					parent = status.MPI_SOURCE;
					parentSet = true;
				} else {
					MPI_Recv(&value, 1, MPI_INT, parent, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				}
				tag = status.MPI_TAG;
				switch (tag) {
					case TAG_ACTIVE: {
						int helperInfo[3];
						MPI_Recv(&(helperInfo[0]), 3, MPI_INT, parent, TAG_HELPER, MPI_COMM_WORLD, &status);
						height = helperInfo[0];
						width = helperInfo[1];
						maxVal = helperInfo[2];
						receivedPixels = (int *) calloc(height * width, sizeof(int));
						MPI_Recv(&(receivedPixels[0]), height * width, MPI_INT, parent, MPI_ANY_TAG, MPI_COMM_WORLD,
						         &status);
						
						filterTag = status.MPI_TAG;
						
						pixelMatrix = (int **) calloc(height, sizeof(int *));
						for (int j = 0; j < height; j++)
							pixelMatrix[j] = (int *) calloc(width, sizeof(int));
						/*
						 * Nodurile frunza trebuie sa reconstruiasca matricea de
						 * pixeli pentru a aplica filtrele.
						 */
						for (int i = 0; i < height; i++)
							for (int j = 0; j < width; j++)
								pixelMatrix[i][j] = receivedPixels[i * width + j];
						
						free(receivedPixels);
						receivedPixels = NULL;
						
						int sobelMatrix[3][3] = {{1, 0, -1},
						                         {2, 0, -2},
						                         {1, 0, -1}};
						int meanRemovalMatrix[3][3] = {{-1, -1, -1},
						                               {-1, 9,  -1},
						                               {-1, -1, -1}};
						int convolutionMatrix[3][3];
						int factor;
						int deplasament;
						int **newPixelMatrix = (int **) calloc(height - 2, sizeof(int *));
						for (int j = 0; j < height - 2; j++)
							newPixelMatrix[j] = (int *) calloc(width - 2, sizeof(int));
						
						if (filterTag == TAG_SOBEL) {
							for (int i = 0; i < 3; i++)
								for (int j = 0; j < 3; j++)
									convolutionMatrix[i][j] = sobelMatrix[i][j];
							factor = 1;
							deplasament = 127;
						} else {
							for (int i = 0; i < 3; i++)
								for (int j = 0; j < 3; j++)
									convolutionMatrix[i][j] = meanRemovalMatrix[i][j];
							factor = 1;
							deplasament = 0;
						}
						
						int sum = 0;
						for (int i = 1; i < height - 1; i++)
							for (int j = 1; j < width - 1; j++) {
								sum = 0;
								for (int k = i - 1, l = 0; k <= i + 1; k++, l++)
									for (int m = j - 1, n = 0; m <= j + 1; m++, n++)
										sum += pixelMatrix[k][m] * convolutionMatrix[l][n];
								int auxiliary = (sum / factor) + deplasament;
								if (auxiliary > maxVal)
									auxiliary = maxVal;
								if (auxiliary < 0)
									auxiliary = 0;
								newPixelMatrix[i - 1][j - 1] = auxiliary;
							}
						
						height -= 2;
						width -= 2;
						nrLinesProcessed += height;
						newPixels = (int *) calloc(height * width, sizeof(int));
						
						for (int i = 0; i < height; i++) {
							for (int j = 0; j < width; j++) {
								newPixels[i * width + j] = newPixelMatrix[i][j];
							}
							
						}
						
						for (int j = 0; j < height + 2; j++) {
							free(pixelMatrix[j]);
							pixelMatrix[j] = NULL;
						}
						free(pixelMatrix);
						pixelMatrix = NULL;
						
						for (int j = 0; j < height; j++) {
							free(newPixelMatrix[j]);
							newPixelMatrix[j] = NULL;
						}
						free(newPixelMatrix);
						newPixelMatrix = NULL;
						
						MPI_Send(&height, 1, MPI_INT, parent, TAG_HELPER, MPI_COMM_WORLD);
						MPI_Send(&width, 1, MPI_INT, parent, TAG_HELPER, MPI_COMM_WORLD);
						MPI_Send(&(newPixels[0]), height * width, MPI_INT, parent, filterTag, MPI_COMM_WORLD);
						free(newPixels);
						newPixels = NULL;
					}
						break;
					case TAG_TERMINATE:
						MPI_Send(&nrLinesProcessed, 1, MPI_INT, parent, TAG_NUMBER, MPI_COMM_WORLD);
						break;
					
				}
			}
		}
		/*
		 * Aici se termina blocul de cod specific unui nod frunza.
		 */
	}
	MPI_Finalize();
	topologyFile.close();
	return 0;
}

Image readImageFromFile(std::string inputImageName) {
	Image image;
	std::ifstream inputImage(inputImageName.c_str());
	
	if (!inputImage) {
		printf("Eroare la deschiderea fisierului %s!\n", inputImageName.c_str());
		exit(-1);
	}
	
	char c;
	std::stringstream auxiliaryHeader;
	int posi = 1, posj = 1;
	std::string line;
	std::vector<std::string> header;
	std::vector<char> whitespaces;
	std::string info;
	std::string pixel;
	/*
	 * In cazul in care intre date sunt 2 sau mai multe whitespaceuri.
	 */
	bool encounteredWhitespace = false;
	bool initialise = false;
	do {
		
		/*
		 * Headerul unei poze contine 4 elemente importante: numele pozei,
		 * lungimea, latimea si valoarea maxima pe care o pot avea
		 * intensitatile pixelilor. Odata ce am citit aceste 4 informatii
		 * restul fisierului este reprezentat de matricea de pixeli.
		 */
		if (header.size() < 4) {
			c = inputImage.peek();
			if (c == '#') {
				/*
				 * Conform standardului PGM, un comentariu incepe cu #
				 * si se termina cu \n.
				 */
				std::getline(inputImage, line);
				/*
				 * std::getline citeste caracterul newline, dar nu il scrie in
				 * fisier, asa ca trebuie sa il scriu manual.
				 */
				auxiliaryHeader << line << "\n";
			} else {
				/*
				 * In poza prelucrata trebuie sa scriem headerul nemodificat
				 * ceea ce implica si pastrarea whitespace-urilor. Daca
				 * intalnesc un caracter ce nu e whitespace, il concatenez
				 * la stringul info, altfel scriu stringul info in fisier
				 * urmat de whitespace. Trebuie sa tinem cont si de faptul
				 * ca doua cuvinte pot fi separate de mai mult de un
				 * caracter whitespace.
				 */
				c = inputImage.get();
				if (!isspace(c)) {
					info += c;
					if (encounteredWhitespace)
						encounteredWhitespace = false;
				} else {
					/*
					 * Daca acesta este primul caracter whitespace intalnit de la
					 * terminarea cuvantului, atunci scriem informatia relevanta
					 * in header, si indicam faptul ca am intalnit deja un caracter
					 * whitespace, pentru a nu duplica continutul headerului.
					 */
					if (!encounteredWhitespace) {
						auxiliaryHeader << info;
						header.push_back(info);
						info.clear();
						encounteredWhitespace = true;
					}
					auxiliaryHeader << c;
				}
			}
		} else {
			if (!initialise) {
				image.height = atoi(header[2].c_str()) + 2;
				image.width = atoi(header[1].c_str()) + 2;
				image.maxVal = atoi(header[3].c_str());
				/*
				 * Functia calloc initializeaza intregul spatiu alocat cu zero,
				 * deci cum matricea este formata doar din zerouri, putem considera
				 * ca este deja bordata, avand grija sa mentinem aceste borduri
				 * atunci cand completam matricea cu pixelii din fisier.
				 */
				image.pixelMatrix = (int **) calloc(image.height, sizeof(int *));
				for (int j = 0; j < image.height; j++)
					image.pixelMatrix[j] = (int *) calloc(image.width, sizeof(int));
				initialise = true;
			}
			
			if (posj == image.width - 1) {
				posj = 1;
				posi++;
			}
			
			c = inputImage.get();
			/*
			 * Deoarece va fi nevoie sa rescriem caracterele whitespace
			 * ce separa pixelii, in timp ce completez matricea de pixeli
			 * retin intr-un vector caracterele whitespace. Cum pixelii
			 * ar putea fi separati de mai multe caractere whitespace,
			 * punem si un caracter delimitator.
			 */
			char delimiter = '|';
			if (!isspace(c)) {
				pixel += c;
				if (encounteredWhitespace) {
					encounteredWhitespace = false;
					image.whitespaces.push_back(delimiter);
				}
			} else {
				if (!encounteredWhitespace) {
					image.pixelMatrix[posi][posj++] = atoi(pixel.c_str());
					pixel.clear();
					encounteredWhitespace = true;
				}
				image.whitespaces.push_back(c);
			}
		}
	} while (c != EOF);
	image.header = auxiliaryHeader.str();
	auxiliaryHeader.clear();
	inputImage.close();
	return image;
}

void writeImageToFile(std::string outputImageName, Image image) {
	std::ofstream outputImage(outputImageName.c_str());
	outputImage << image.header;
	std::vector<char>::iterator it = image.whitespaces.begin();
	it++;
	for (int i = 0; i < image.pixelVector.size(); i++) {
		outputImage << image.pixelVector[i];
		while (*it != '|') {
			outputImage << *it;
			it++;
		}
		it++;
	}
	image.whitespaces.clear();
	outputImage.close();
}

void sendDataToChildren(Image image, int helperInfo[3], int parent, int filterTag, bool isRoot, int *receivedPixels,
                        std::vector<int> neighbours) {
	int nrLinesToSend;
	int rest;
	int line = 0;
	bool flag;
	int nrNeighbours;
	int *pixels;
	int value;
	
	if (isRoot)
		nrNeighbours = neighbours.size();
	else
		/*
		 * Parintele nu trebuie considerat atunci cand impartim
		 * blocul primit in blocuri mai mici ce vor fi trimise
		 * copiilor, in cazul nodurilor intermediare.
		 */
		nrNeighbours = neighbours.size() - 1;
	
	int height = helperInfo[0];
	int width = helperInfo[1];
	int maxVal = helperInfo[2];
	
	/*
	 * Daca sunt mai putine linii decat copii, atunci se trimite cate o
	 * linie la primii height - 2 copii.
	 */
	if (height - 2 < nrNeighbours) {
		flag = false;
		/*
		 * Fiecare vecin primeste doar 3 linii, adica linia corespunzatoare
		 * si cele doua margini.
		 */
		nrLinesToSend = 3;
	} else {
		flag = true;
		
			/*
			* Scadem 2 din numarul de linii, deoarece 2 linii au fost adaugate
			* prin bordarea matricei cu zerouri, iar ele nu trebuie luate in
			* considerare la impartirea numarului de linii catre copii. De asemenea,
			* este nevoie sa scadem 2 si la nodurile intermediare, pentru a nu lua
			* in considerare la impartirea numarului de linii marginile de sus si 
			* jos ce au fost primite.
			* In schimb adun 2, deoarece fiecare bloc trimis contine si marginile
			* de sus si jos.
			*/
		nrLinesToSend = (height - 2) / nrNeighbours + 2;
		rest = (height - 2) % nrNeighbours;
	}
	
	/*
	 * Tratam cele doua cazuri in mod similar, facand distinctia intre ele prin
	 * intermediul variabilei flag.
	 */
	for (int i = 0, nrLinesSent = 0;
	     (i < neighbours.size() && flag) || (nrLinesSent < height - 2 && !flag); i++) {
		if (neighbours[i] != parent) {
			
			/*
			 * Daca am ajuns la ultimul vecin si sunt in cazul normal ( nrBlocuri
			 * > nrCopii ), adaug restul la numarul de linii ce trebuie trimise.
			 */
			if (i == neighbours.size() - 1 && flag)
				nrLinesToSend += rest;
			
			pixels = (int *) calloc(nrLinesToSend * width, sizeof(int));
			if (isRoot) {
				/*
				 * Nu pot trimite matrice prin MPI, asa ca o sa creez un
				 * vector format din liniile blocului de pixeli ce trebuie
				  * trimis la copii.
				 */
				for (int j = line, x = 0; j < line + nrLinesToSend; j++, x++)
					for (int k = 0; k < image.width; k++)
						pixels[x * image.width + k] = image.pixelMatrix[j][k];
			} else {
				for (int j = line * width, x = 0; j < (line + nrLinesToSend) * width; j++, x++)
					pixels[x] = receivedPixels[j];
			}
			
			line = line + nrLinesToSend - 2;
			MPI_Send(&value, 1, MPI_INT, neighbours[i], TAG_ACTIVE, MPI_COMM_WORLD);
			helperInfo[0] = nrLinesToSend;
			helperInfo[1] = width;
			helperInfo[2] = maxVal;
			MPI_Send(&(helperInfo[0]), 3, MPI_INT, neighbours[i], TAG_HELPER, MPI_COMM_WORLD);
			MPI_Send(&(pixels[0]), nrLinesToSend * width, MPI_INT, neighbours[i], filterTag,
			         MPI_COMM_WORLD);
			
			nrLinesSent++;
			free(pixels);
			pixels = NULL;
		}
	}
}

void receiveDataFromChildren(Image &image, int helperInfo[3], int parent, int filterTag, bool isRoot,
                             std::vector<int> neighbours) {
	
	int height;
	int width;
	int *receivedPixels;
	bool flag;
	int nrNeighbours;
	int nrReceivedLines = 0;
	std::vector<int> collectPixels;
	MPI_Status status;
	
	if (isRoot) {
		nrNeighbours = neighbours.size();
		height = image.height;
		width = image.width;
	} else {
		nrNeighbours = neighbours.size() - 1;
		height = helperInfo[0];
	}
	
	if (height - 2 < nrNeighbours)
		flag = false;
	else
		flag = true;
	
	
	for (int i = 0, nrLinesSent = 0;
	     (i < neighbours.size() && flag) || (nrLinesSent < height - 2 && !flag); i++) {
		
		if (neighbours[i] != parent) {
			MPI_Recv(&height, 1, MPI_INT, neighbours[i], TAG_HELPER, MPI_COMM_WORLD, &status);
			MPI_Recv(&width, 1, MPI_INT, neighbours[i], TAG_HELPER, MPI_COMM_WORLD, &status);
			receivedPixels = (int *) calloc(height * width, sizeof(int));
			MPI_Recv(&(receivedPixels[0]), height * width, MPI_INT, neighbours[i], MPI_ANY_TAG,
			         MPI_COMM_WORLD, &status);
			
			if (isRoot) {
				for (int j = 0; j < height * width; j++)
					image.pixelVector.push_back(receivedPixels[j]);
			} else {
				for (int j = 0; j < height * width; j++)
					collectPixels.push_back(receivedPixels[j]);
				nrReceivedLines += height;				
			}
			
			nrLinesSent++;
			free(receivedPixels);
			receivedPixels = NULL;
		}
		
	}
	
	/*
	 * Daca nodul este intermediar, atunci trimite la parinte numarul de linii
	 * pe care le-a primit, latimea si vectorul ce contine vectorii de pixeli
	 * primiti de la copii concatenati.
	 */
	if (!isRoot) {
		MPI_Send(&nrReceivedLines, 1, MPI_INT, parent, TAG_HELPER, MPI_COMM_WORLD);
		MPI_Send(&width, 1, MPI_INT, parent, TAG_HELPER, MPI_COMM_WORLD);
		int *vector = collectPixels.data();
		MPI_Send(&(vector[0]), nrReceivedLines * width, MPI_INT, parent, filterTag, MPI_COMM_WORLD);
		collectPixels.clear();
	}
	
}

void receiveStatisticsFromChildren(std::vector<int> &ranks, std::vector<int> &statistics,
                                   std::vector<int> neighbours, int rank, int parent) {
	ranks.push_back(rank);
	statistics.push_back(0);
	int nr;
	MPI_Status status;
	
	/*
	 * Tratam radacina ca un nod intermediar ce are parintele -1.
	 */
	for (int i = 0; i < neighbours.size(); i++)
		if (neighbours[i] != parent) {
			MPI_Recv(&nr, 1, MPI_INT, neighbours[i], MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			/*
			 * Trebuie sa facem distinctia in functie de TAG intre nodurile care
			 * trimit informatii. Nodurile frunza trimit doar numarul de linii
			 * procesate, insa nodurile intermediare trimit 2 vectori ce contin
			 * rankurile si statisticile concatenate.
			 */
			if (status.MPI_TAG == TAG_NUMBER) {
				statistics.push_back(nr);
				ranks.push_back(neighbours[i]);
			} else {
				int *receivedRanks = (int *) calloc(nr, sizeof(int));
				MPI_Recv(&(receivedRanks[0]), nr, MPI_INT, neighbours[i], MPI_ANY_TAG,
				         MPI_COMM_WORLD, &status);
				int *receivedStatistics = (int *) calloc(nr, sizeof(int));
				MPI_Recv(&(receivedStatistics[0]), nr, MPI_INT, neighbours[i], MPI_ANY_TAG,
				         MPI_COMM_WORLD, &status);
				for (int i = 0; i < nr; i++) {
					ranks.push_back(receivedRanks[i]);
					statistics.push_back(receivedStatistics[i]);
				}
				
			}
			
		}
}
