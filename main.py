from time import sleep
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
import numpy
import pyaudio
import pyperclip
import rumps
import threading
import torch

LANGUAGE_CODES = {
    "Acehnese Arabic": "ace_Arab",
    "Acehnese Latin": "ace_Latn",
    "Mesopotamian Arabic": "acm_Arab",
    "Ta'izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Afrikaans": "afr_Latn",
    "South Levantine Arabic": "ajp_Arab",
    "Akan": "aka_Latn",
    "Amharic": "amh_Ethi",
    "North Levantine Arabic": "apc_Arab",
    "Modern Standard Arabic": "arb_Arab",
    "Modern Standard Arabic Romanized": "arb_Latn",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",
    "Assamese": "asm_Beng",
    "Asturian": "ast_Latn",
    "Awadhi": "awa_Deva",
    "Central Aymara": "ayr_Latn",
    "South Azerbaijani": "azb_Arab",
    "North Azerbaijani": "azj_Latn",
    "Bashkir": "bak_Cyrl",
    "Bambara": "bam_Latn",
    "Balinese": "ban_Latn",
    "Belarusian": "bel_Cyrl",
    "Bemba": "bem_Latn",
    "Bengali": "ben_Beng",
    "Bhojpuri": "bho_Deva",
    "Banjar Arabic": "bjn_Arab",
    "Banjar Latin": "bjn_Latn",
    "Standard Tibetan": "bod_Tibt",
    "Bosnian": "bos_Latn",
    "Buginese": "bug_Latn",
    "Bulgarian": "bul_Cyrl",
    "Catalan": "cat_Latn",
    "Cebuano": "ceb_Latn",
    "Czech": "ces_Latn",
    "Chokwe": "cjk_Latn",
    "Central Kurdish": "ckb_Arab",
    "Crimean Tatar": "crh_Latn",
    "Welsh": "cym_Latn",
    "Danish": "dan_Latn",
    "German": "deu_Latn",
    "Southwestern Dinka": "dik_Latn",
    "Dyula": "dyu_Latn",
    "Dzongkha": "dzo_Tibt",
    "Greek": "ell_Grek",
    "English": "eng_Latn",
    "Esperanto": "epo_Latn",
    "Estonian": "est_Latn",
    "Basque": "eus_Latn",
    "Ewe": "ewe_Latn",
    "Faroese": "fao_Latn",
    "Fijian": "fij_Latn",
    "Finnish": "fin_Latn",
    "Fon": "fon_Latn",
    "French": "fra_Latn",
    "Friulian": "fur_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
    "Scottish Gaelic": "gla_Latn",
    "Irish": "gle_Latn",
    "Galician": "glg_Latn",
    "Guarani": "grn_Latn",
    "Gujarati": "guj_Gujr",
    "Haitian Creole": "hat_Latn",
    "Hausa": "hau_Latn",
    "Hebrew": "heb_Hebr",
    "Hindi": "hin_Deva",
    "Chhattisgarhi": "hne_Deva",
    "Croatian": "hrv_Latn",
    "Hungarian": "hun_Latn",
    "Armenian": "hye_Armn",
    "Igbo": "ibo_Latn",
    "Ilocano": "ilo_Latn",
    "Indonesian": "ind_Latn",
    "Icelandic": "isl_Latn",
    "Italian": "ita_Latn",
    "Javanese": "jav_Latn",
    "Japanese": "jpn_Jpan",
    "Kabyle": "kab_Latn",
    "Jingpho": "kac_Latn",
    "Kamba": "kam_Latn",
    "Kannada": "kan_Knda",
    "Kashmiri Arabic": "kas_Arab",
    "Kashmiri Devanagari": "kas_Deva",
    "Georgian": "kat_Geor",
    "Central Kanuri Arabic": "knc_Arab",
    "Central Kanuri Latin": "knc_Latn",
    "Kazakh": "kaz_Cyrl",
    "Kabiyè": "kbp_Latn",
    "Kabuverdianu": "kea_Latn",
    "Khmer": "khm_Khmr",
    "Kikuyu": "kik_Latn",
    "Kinyarwanda": "kin_Latn",
    "Kyrgyz": "kir_Cyrl",
    "Kimbundu": "kmb_Latn",
    "Northern Kurdish": "kmr_Latn",
    "Kikongo": "kon_Latn",
    "Korean": "kor_Hang",
    "Lao": "lao_Laoo",
    "Ligurian": "lij_Latn",
    "Limburgish": "lim_Latn",
    "Lingala": "lin_Latn",
    "Lithuanian": "lit_Latn",
    "Lombard": "lmo_Latn",
    "Latgalian": "ltg_Latn",
    "Luxembourgish": "ltz_Latn",
    "Luba-Kasai": "lua_Latn",
    "Ganda": "lug_Latn",
    "Luo": "luo_Latn",
    "Mizo": "lus_Latn",
    "Standard Latvian": "lvs_Latn",
    "Magahi": "mag_Deva",
    "Maithili": "mai_Deva",
    "Malayalam": "mal_Mlym",
    "Marathi": "mar_Deva",
    "Minangkabau Arabic ": "min_Arab",
    "Minangkabau Latin": "min_Latn",
    "Macedonian": "mkd_Cyrl",
    "Plateau Malagasy": "plt_Latn",
    "Maltese": "mlt_Latn",
    "Meitei Bengali": "mni_Beng",
    "Halh Mongolian": "khk_Cyrl",
    "Mossi": "mos_Latn",
    "Maori": "mri_Latn",
    "Burmese": "mya_Mymr",
    "Dutch": "nld_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Nepali": "npi_Deva",
    "Northern Sotho": "nso_Latn",
    "Nuer": "nus_Latn",
    "Nyanja": "nya_Latn",
    "Occitan": "oci_Latn",
    "West Central Oromo": "gaz_Latn",
    "Odia": "ory_Orya",
    "Pangasinan": "pag_Latn",
    "Eastern Panjabi": "pan_Guru",
    "Papiamento": "pap_Latn",
    "Western Persian": "pes_Arab",
    "Polish": "pol_Latn",
    "Portuguese": "por_Latn",
    "Dari": "prs_Arab",
    "Southern Pashto": "pbt_Arab",
    "Ayacucho Quechua": "quy_Latn",
    "Romanian": "ron_Latn",
    "Rundi": "run_Latn",
    "Russian": "rus_Cyrl",
    "Sango": "sag_Latn",
    "Sanskrit": "san_Deva",
    "Santali": "sat_Olck",
    "Sicilian": "scn_Latn",
    "Shan": "shn_Mymr",
    "Sinhala": "sin_Sinh",
    "Slovak": "slk_Latn",
    "Slovenian": "slv_Latn",
    "Samoan": "smo_Latn",
    "Shona": "sna_Latn",
    "Sindhi": "snd_Arab",
    "Somali": "som_Latn",
    "Southern Sotho": "sot_Latn",
    "Spanish": "spa_Latn",
    "Tosk Albanian": "als_Latn",
    "Sardinian": "srd_Latn",
    "Serbian": "srp_Cyrl",
    "Swati": "ssw_Latn",
    "Sundanese": "sun_Latn",
    "Swedish": "swe_Latn",
    "Swahili": "swh_Latn",
    "Silesian": "szl_Latn",
    "Tamil": "tam_Taml",
    "Tatar": "tat_Cyrl",
    "Telugu": "tel_Telu",
    "Tajik": "tgk_Cyrl",
    "Tagalog": "tgl_Latn",
    "Thai": "tha_Thai",
    "Tigrinya": "tir_Ethi",
    "Tamasheq Latin": "taq_Latn",
    "Tamasheq Tifinagh": "taq_Tfng",
    "Tok Pisin": "tpi_Latn",
    "Tswana": "tsn_Latn",
    "Tsonga": "tso_Latn",
    "Turkmen": "tuk_Latn",
    "Tumbuka": "tum_Latn",
    "Turkish": "tur_Latn",
    "Twi": "twi_Latn",
    "Central Atlas Tamazight": "tzm_Tfng",
    "Uyghur": "uig_Arab",
    "Ukrainian": "ukr_Cyrl",
    "Umbundu": "umb_Latn",
    "Urdu": "urd_Arab",
    "Northern Uzbek": "uzn_Latn",
    "Venetian": "vec_Latn",
    "Vietnamese": "vie_Latn",
    "Waray": "war_Latn",
    "Wolof": "wol_Latn",
    "Xhosa": "xho_Latn",
    "Eastern Yiddish": "ydd_Hebr",
    "Yoruba": "yor_Latn",
    "Yue Chinese": "yue_Hant",
    "Chinese Simplified": "zho_Hans",
    "Chinese Traditional": "zho_Hant",
    "Standard Malay": "zsm_Latn",
    "Zulu": "zul_Latn",
}

fs=16000

transcript = "Transcript"
transcript_to_english = "Transcript -> ENG"
transcript_to_spanish = "Translate -> ESP"
transcript_to_russian = "Translate -> RUS"

class Recorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.frames_per_buffer = 1024

    def start(self):
        self.recording = True
        thread = threading.Thread(target=self._record_impl)
        thread.start()

    def _record_impl(self):
        self.recording = True
        print("recording...")
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=fs,
                        frames_per_buffer=self.frames_per_buffer,
                        input=True)
        self.frames = []
        while self.recording:
            data = stream.read(self.frames_per_buffer)
            self.frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("recordering stopped")

    def stop(self):
        self.recording = False
        sleep(0.5)
        return self.frames

class RecorderApp(rumps.App):
    def __init__(self, transcriptor_model, translator_model):
        super(RecorderApp, self).__init__("🔵", menu=["Transcript"])
        self.recorder = Recorder()
        print('loading model', transcriptor_model)
        self.transcriptor=pipeline(
            "automatic-speech-recognition",
            torch_dtype=torch.float16,
            model=transcriptor_model,
            device="mps",
            model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
        )
        print('loading model', translator_model)
        self.en2ru=pipeline(
            "translation",
            model=translator_model,
            src_lang=LANGUAGE_CODES['English'],
            tgt_lang=LANGUAGE_CODES['Russian'],
        )
        self.en2es=pipeline(
            "translation",
            model=translator_model,
            src_lang=LANGUAGE_CODES['English'],
            tgt_lang=LANGUAGE_CODES['Spanish'],
        )
        print('models loaded')

    def start(self):
        self.title='🔴'
        self.recorder.start()

    def stop(self):
        return self.recorder.stop()

    def transcribe(self, frames, args):
        print("transcribing...")
        audio_data = numpy.frombuffer(b''.join(frames), dtype=numpy.int16)
        audio_data_fp32 = audio_data.astype(numpy.float32) / 32768.0
        text = self.transcriptor(
            audio_data_fp32,
            chunk_length_s=30,
            batch_size=24,
            return_timestamps=True,
            generate_kwargs=args
        )
        return text['text'].strip()

    @rumps.clicked(transcript)
    def transcript(self, _):
        if self.recorder.recording:
            frames = self.stop()
            text = self.transcribe(frames, {})
            print(text)
            pyperclip.copy(text)
            self.title='🔵'
            self.menu[transcript].title = transcript
        else:
            self.start()
            self.menu[transcript].title = '⏺ ' + transcript

    @rumps.clicked(transcript_to_english)
    def transcript_to_english(self, _):
        if self.recorder.recording:
            frames = self.stop()
            text = self.transcribe(frames, {"task": "translate"})
            print(text)
            pyperclip.copy(text)
            self.title='🔵'
            self.menu[transcript_to_english].title = transcript_to_english
        else:
            self.start()
            self.menu[transcript_to_english].title = '⏺ ' + transcript_to_english

    @rumps.clicked(transcript_to_spanish)
    def transcript_to_spanish(self, _):
        if self.recorder.recording:
            frames = self.stop()
            text = self.transcribe(frames, {"task": "translate"})
            print(text)
            print("translating...")
            text = self.en2es(text)[0]['translation_text']
            print(text)
            pyperclip.copy(text)
            self.title='🔵'
            self.menu[transcript_to_spanish].title = transcript_to_spanish
        else:
            self.start()
            self.menu[transcript_to_spanish].title = '⏺ ' + transcript_to_spanish

    @rumps.clicked(transcript_to_russian)
    def transcript_to_russian(self, _):
        if self.recorder.recording:
            frames = self.stop()
            text = self.transcribe(frames, {"task": "translate"})
            print(text)
            print("translating...")
            text = self.en2ru(text)[0]['translation_text']
            print(text)
            pyperclip.copy(text)
            self.title='🔵'
            self.menu[transcript_to_russian].title = transcript_to_russian
        else:
            self.start()
            self.menu[transcript_to_russian].title = '⏺ ' + transcript_to_russian

if __name__ == "__main__":
    app = RecorderApp("openai/whisper-large-v3", "facebook/nllb-200-distilled-600M")
    app.run()
