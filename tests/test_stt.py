# tests/test_stt.py
from core.services.stt import transcribe_audio

def test_transcription_activities_grade1():
    # Activities
    assert "Vytvoř hovor s Janou Malinovou o projektu Nová kampaň.".lower() == transcribe_audio("data/examples/audio/activity1_normal_grade1.mp3").strip().lower()
    assert "Pošli e-mail Petrovi Novotnému s poslední verzí nabídky pro Innotec s.r.o.".lower() == transcribe_audio("data/examples/audio/activity2_normal_grade1.mp3", correct=False).strip().lower()
    assert "Naplánuj schůzku s firmou Plas na pátek ve 14:00.".lower() == transcribe_audio("data/examples/audio/activity3_normal_grade1.mp3", correct=False).strip().lower()
    assert "Připomeň mi kontaktovat firmu Globus příští týden.".lower() == transcribe_audio("data/examples/audio/activity4_normal_grade1.mp3", correct=False).strip().lower()
    assert "Jaká schůzky mám na tento týden?".lower() == transcribe_audio("data/examples/audio/activity5_normal_grade1.mp3", correct=False).strip().lower()
    assert "V kolik hodin je dnes schůzka s firmou Alefbet.".lower() == transcribe_audio("data/examples/audio/activity6_normal_grade1.mp3", correct=False).strip().lower()

def test_transcription_contacts_grade1():
    # Contacts
    assert "Přidej nový kontakt jménem Jan Novák z firmy Acmark.".lower() == transcribe_audio("data/examples/audio/contact1_normal_grade1.mp3").strip().lower()
    assert "Změň telefonní číslo u Petra Dvořáka na 777 123 456.".lower() == transcribe_audio("data/examples/audio/contact2_normal_grade1.mp3", correct=False).strip().lower()
    assert "Zobraz kontakt Jany Svobodové.".lower() == transcribe_audio("data/examples/audio/contact3_normal_grade1.mp3", correct=False).strip().lower()
    assert "Smaž kontakt Tomáš Horák.".lower() == transcribe_audio("data/examples/audio/contact4_normal_grade1.mp3", correct=False).strip().lower()
    assert "Zobraz kontakt Martina Kuáka z firmy Iveco.".lower() == transcribe_audio("data/examples/audio/contact5_normal_grade1.mp3", correct=False).strip().lower()

# - „Vytvoř novou nabídku s firmou Globex na produkty alu kola R17, 4 kusy“
# - „Přesuň poslední nabídku s firmou Janek a.s. do stavu vyhráli jsme.“
# - „Jaké nabídky se mají uzavřít tento měsíc?“
# - „Jaký je odhadovaný příjem na příští čtvrtletí?“
def test_transcription_opportunities_grade2():
    # Opportunities
    assert "Vytvoř novou nabídku s firmou Globex na produkty alu kola R17, 4 kusy.".lower() == transcribe_audio("data/examples/audio/opportunites1_correction_grade2.mp3").strip().lower()
    assert "Přesuň poslední nabídku s firmou Janek a.s. do stavu vyhráli jsme.".lower() == transcribe_audio("data/examples/audio/opportunites2_correction_grade2.mp3").strip().lower()
    assert "Jaké nabídky se mají uzavřít tento měsíc?".lower() == transcribe_audio("data/examples/audio/opportunites3_correction_grade2.mp3").strip().lower()
    assert "Jaký je odhadovaný příjem na příští čtvrtletí?".lower() == transcribe_audio("data/examples/audio/opportunites4_correction_grade2.mp3").strip().lower()
    # correction
    assert "Vytvoř novou nabídku s firmou Globex na produkty alu kola R17, 4 kusy.".lower() == transcribe_audio("data/examples/audio/opportunites1_correction_grade2.mp3").strip().lower()
