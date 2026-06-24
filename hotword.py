import numpy as np
from scipy.io import wavfile
from scipy import signal as sig      
import matplotlib.pyplot as plt
import IPython


def promijeni_rate(wav, stari_rate, novi_rate):
    broj_odbiraka = int(len(wav) * novi_rate / stari_rate)
    return sig.resample(wav, broj_odbiraka)


def ucitaj(putanja: str, ciljni_rate=16000):
    rate, wav = wavfile.read(putanja)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = promijeni_rate(wav, rate, ciljni_rate)
    rate = ciljni_rate
    print(f"frekvencija odabiranja - ", rate)
    print(f"broj odabiraka", wav.shape)
    return rate, wav

def filtriraj_puter(snimak, rate, donji_prag=100, gornji_prag=7000, red=5):#ovi pragove sluze fakticki da ti vidis kju frekvenciju propustas a koju ne 

    b, a = sig.butter(red, [donji_prag, gornji_prag], btype='band', fs=rate)#inicijalizujes parametre za taj filter jer postoje vise stepena tog filtera
    filtriran = sig.lfilter(b, a, snimak)#pozivas filter sa parametrima da procisti signal

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex="col", figsize=(10, 4), sharey="col")
    ax1.specgram(filtriran, Fs=rate)
    ax2.magnitude_spectrum(filtriran[:int(3*rate)], Fs=rate, scale='dB')
    ax2.magnitude_spectrum(snimak[:int(3*rate)], Fs=rate, scale='dB')
    ax2.set_ylim(-120, 70)
    ax2.grid()
    plt.show()
    return filtriran

def uklanjanje_tisine(wav, rate, prag_trenutni=0.02, frejm_gledanja=30):
    duzina_frejma = int(rate * frejm_gledanja / 1000)
    broj_frejmova = len(wav) // duzina_frejma
    energije = []
    for i in range(broj_frejmova):
        pocetak = i * duzina_frejma
        kraj = (i + 1) * duzina_frejma
        frejm = np.zeros(duzina_frejma)
        for j in range(duzina_frejma):
            frejm[j] = wav[pocetak + j]
        frejm = frejm.astype(np.float64)
        energija = np.sum(frejm ** 2)
        snaga = energija / len(frejm)
        efektivna = np.sqrt(snaga)
        energije.append(efektivna)
    maks_energija = max(energije)

    rezultat = []
    for i, energija in enumerate(energije):
        if energija >= prag_trenutni * maks_energija:
            pocetak = i * duzina_frejma
            for j in range(duzina_frejma):
                rezultat.append(wav[pocetak + j])


    return np.array(rezultat)
def predobradi(putanja: str, prikazi=True):
    rate, wav = ucitaj(putanja)
    filtriran = filtriraj_puter(wav, rate)
    bez_tisine = uklanjanje_tisine(filtriran, rate)

    print(f"len sa tision: {len(filtriran)}")
    print(f"len bez tisine: {len(bez_tisine)}")

    if prikazi:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex="col", figsize=(10, 4), sharey="col")
        ax1.specgram(bez_tisine, Fs=rate)
        ax1.set_title("Spektrogram (final)")
        ax2.magnitude_spectrum(bez_tisine, Fs=rate, scale='dB')
        ax2.set_ylim(-120, 70)
        ax2.grid()
        plt.show()
        IPython.display.Audio(data=bez_tisine, rate=rate)

    return rate, bez_tisine
def DTW(spektar_hotword, spektar_recenica):
    visina, sirina=spektar_hotword.shape[1],spektar_recenica.shape[1]#rec, prozor
    frekvencije=spektar_hotword.shape[0]
    matrica =[]
    matrica=np.zeros((visina, sirina))
    for i in range(visina):
        for j in range (sirina):
            razlika=0
            for k in range(frekvencije):
                razlika+=(spektar_hotword[k,i]-spektar_recenica[k,j])**2
            matrica[i,j]=np.sqrt(razlika)
    matrica_put=np.full((visina, sirina), np.inf)
    matrica_put[0,0]=matrica[0,0]
    for i in range (1, visina):
        matrica_put[i,0]=matrica_put[i-1, 0]+matrica[i,0]
    for i in range(1,sirina):
        matrica_put[0, i]=matrica[0,i]+matrica_put[0,i-1]
    for i in range (1, visina):
        for j in range(1, sirina):
            matrica_put[i,j]=matrica[i,j]+min(matrica_put[i-1,j], matrica_put[i-1, j-1], matrica_put[i,j-1])
    return matrica_put[visina-1, sirina-1]/(sirina+visina)#aproksimira distancu jer u zavisnoti od audia imamo vie frejmova
def trazenje_sekunde(hotword, recenica, rate_hotword,rate_recenica, prag=100):
    duzina_prozora=len(hotword)
    korak=int(0.1*rate_recenica)
    distance=[]
    vremena=[]
    spektar_hotword=sig.spectrogram(hotword,rate_hotword)[2]
    for i in range (0, len(recenica)-duzina_prozora, korak):
        frame=np.zeros(duzina_prozora)
        for j in range(duzina_prozora):
            frame[j]=recenica[i + j]
        spektar_frame=sig.spectrogram(frame,rate_recenica)[2]
        d=DTW(spektar_hotword, spektar_frame)
        distance.append(d)
        vremena.append(i/rate_recenica)
    return vremena, distance
def rezultat(vreme, distance):
    minimun=distance[0]
    trenutak=-1
    for i in range (len(distance)):
        if minimun>distance[i]:
            minimun=distance[i]
            trenutak=vreme[i]
    return trenutak
if __name__ == '__main__':
    rate1, wav1 = predobradi("hotword-mono.wav")
    rate2, wav2 = ucitaj("recenia-mono.wav")
    filtirana_recenica=filtriraj_puter(wav2, rate2)
    
    vreme, distance= trazenje_sekunde(wav1,filtirana_recenica, rate1, rate2)
    print ("Hotword je detektovan u trenutku", rezultat(vreme, distance))
   