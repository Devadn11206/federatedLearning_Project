"""Phase 4 Test Suite: Clinical Dashboard & Automated Reporting Verification.

Verifies:
1. app/dashboard.py syntax, imports, and data loaders.
2. reports/generate_final_report.py execution and artifact validation (Markdown and LaTeX).
3. Live cryptographic audit ledger verification through the dashboard integration.
"""

import os
import sys
import json
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from security.audit_ledger import EUAIActAuditLedger


def test_dashboard_file_and_imports():
    """Verify that app/dashboard.py exists and can be imported/parsed without error."""
    dashboard_path = PROJECT_ROOT / "app" / "dashboard.py"
    assert dashboard_path.exists(), "app/dashboard.py does not exist"
    
    # Check syntax by compiling
    with open(dashboard_path, "r", encoding="utf-8") as f:
        source = f.read()
    compiled = compile(source, str(dashboard_path), "exec")
    assert compiled is not None


def test_report_generator_execution():
    """Verify that reports/generate_final_report.py generates valid markdown and latex files."""
    import reports.generate_final_report as rep_gen

    # Execute main generator
    rep_gen.main()

    md_path = PROJECT_ROOT / "reports" / "FedMedVision_Final_Report.md"
    tex_path = PROJECT_ROOT / "reports" / "ablation_table.tex"

    assert md_path.exists(), "FedMedVision_Final_Report.md was not created"
    assert tex_path.exists(), "ablation_table.tex was not created"
    assert md_path.stat().st_size > 500, "Final report markdown is suspiciously small"
    assert tex_path.stat().st_size > 200, "LaTeX ablation table is suspiciously small"

    # Verify content contents
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "FedMedVision" in md_text
    assert "EU Artificial Intelligence Act" in md_text
    assert "Pointing Game Accuracy" in md_text

    with open(tex_path, "r", encoding="utf-8") as f:
        tex_text = f.read()
    assert r"\begin{table*}" in tex_text
    assert "FedMedVision" in tex_text
    assert r"\bottomrule" in tex_text


def test_audit_ledger_read_and_verify():
    """Verify EUAIActAuditLedger read_ledger and verify_chain_integrity work seamlessly."""
    ledger = EUAIActAuditLedger()
    is_valid, tampered_line = ledger.verify_chain_integrity()
    assert is_valid is True, f"Audit ledger chain integrity failed at line {tampered_line}"
    assert tampered_line is None

    records = ledger.read_ledger()
    assert isinstance(records, list)
    assert len(records) > 0, "Audit ledger should contain logged rounds"
    for r in records:
        assert "record_hash" in r
        assert "prev_hash" in r
        assert "round_id" in r


def test_data_artifacts_for_dashboard():
    """Ensure all datasets, split files, and checkpoints expected by dashboard.py exist."""
    required_paths = [
        PROJECT_ROOT / "data" / "splits" / "bbox_eval.csv",
        PROJECT_ROOT / "data" / "splits" / "val_test.csv",
        PROJECT_ROOT / "data" / "splits" / "consortium_stats.json",
        PROJECT_ROOT / "results" / "metrics" / "fl_telemetry.json",
        PROJECT_ROOT / "results" / "metrics" / "ablation_table.json",
        PROJECT_ROOT / "results" / "checkpoints" / "fedmedvision_global.pt",
        PROJECT_ROOT / "results" / "checkpoints" / "isolated_hub_0.pt",
        PROJECT_ROOT / "results" / "checkpoints" / "centralized_oracle.pt",
    ]
    for p in required_paths:
        assert p.exists(), f"Required artifact {p} missing for dashboard"


def test_dicom_windowing_and_pdf_generation():
    """Verify pydicom DICOM windowing logic and reportlab radiology consult PDF generator."""
    import io
    import numpy as np
    from PIL import Image
    from app.dashboard import apply_dicom_windowing, generate_radiology_consult_pdf
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

    # 1. Test DICOM Windowing
    raw_pixels = np.linspace(0, 4000, 224 * 224, dtype=np.float32).reshape((224, 224))
    
    # Auto window
    w_auto = apply_dicom_windowing(raw_pixels, slope=1.0, intercept=0.0)
    assert w_auto.shape == (224, 224)
    assert w_auto.dtype == np.uint8
    assert w_auto.min() == 0 and w_auto.max() == 255

    # Lung window (WW: 1500, WL: -600) with Rescale Intercept
    w_lung = apply_dicom_windowing(raw_pixels, slope=1.0, intercept=-1000.0, window_center=-600.0, window_width=1500.0)
    assert w_lung.shape == (224, 224)
    assert w_lung.dtype == np.uint8

    # Mediastinal window (WW: 400, WL: 40)
    w_med = apply_dicom_windowing(raw_pixels, slope=1.0, intercept=-1000.0, window_center=40.0, window_width=400.0)
    assert w_med.shape == (224, 224)

    # Custom window (WW: 800, WL: 200)
    w_custom = apply_dicom_windowing(raw_pixels, slope=1.0, intercept=0.0, window_center=200.0, window_width=800.0)
    assert w_custom.shape == (224, 224)
    assert w_custom.dtype == np.uint8

    # 2. Test Anisotropic PixelSpacing Letterboxing (Preserves anatomical aspect ratio)
    from app.dashboard import letterbox_pixel_spacing
    aniso_raw = np.ones((100, 200), dtype=np.float32)
    # Row spacing is 2x col spacing -> height scaled by 2 (200x200 square)
    padded = letterbox_pixel_spacing(aniso_raw, (0.2, 0.1))
    assert padded.shape == (200, 200), f"Expected symmetric square (200, 200), got {padded.shape}"

    # 3. Test Synthetic DICOM Creation and Reading with Compression Plugins
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dcm = FileDataset("test_case.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
    dcm.Modality = "CR"
    dcm.PatientID = "ANON_PT_99"
    dcm.ViewPosition = "PA"
    dcm.BitsStored = 16
    dcm.BitsAllocated = 16
    dcm.HighBit = 15
    dcm.PixelRepresentation = 0
    dcm.SamplesPerPixel = 1
    dcm.PhotometricInterpretation = "MONOCHROME2"
    dcm.PixelSpacing = [0.150, 0.150]
    dcm.RescaleSlope = 1.0
    dcm.RescaleIntercept = -1000.0
    dcm.Rows = 224
    dcm.Columns = 224
    dcm.PixelData = (np.random.rand(224, 224) * 4095).astype(np.uint16).tobytes()

    buf = io.BytesIO()
    dcm.save_as(buf)
    dcm_bytes = buf.getvalue()

    # Test Cached DICOM parser
    from app.dashboard import parse_and_cache_dicom
    dcm_parsed = parse_and_cache_dicom(dcm_bytes)
    assert dcm_parsed["modality"] == "CR"
    assert "ANON_" in dcm_parsed["patient_id"]
    assert dcm_parsed["bits_stored"] == 16
    assert dcm_parsed["hu_pixels"].shape == (224, 224)

    # 4. Test ReportLab Single-Page A4 PDF Consult Generation with Physician Notes & Metadata
    dummy_orig = Image.new("RGB", (224, 224), color=(64, 64, 64))
    dummy_overlay = Image.new("RGB", (224, 224), color=(200, 80, 50))

    probs = {
        "Cardiomegaly": 0.6420,
        "Effusion": 0.1210,
        "Infiltration": 0.1530,
        "Pneumonia": 0.0840,
        "No Finding": 0.2100
    }
    uncert = {p: 0.05 for p in probs}

    pdf_bytes = generate_radiology_consult_pdf(
        scan_filename="test_case_001.dcm",
        patient_id="ANON_PT_99",
        active_model_name="FedMedVision Global (fedmedvision_global.pt)",
        model_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        orig_pil=dummy_orig,
        overlay_pil=dummy_overlay,
        primary_finding="Cardiomegaly",
        primary_prob=0.6420,
        pred_entropy=0.2850,
        probs_dict=probs,
        uncert_dict=uncert,
        decision_threshold=0.50,
        rec_action="High clinical priority: Urgent radiologist review and follow-up recommended.",
        dicom_meta=dcm_parsed,
        physician_name="Dr. Eleanor Vance, MD (ABR #84920)",
        physician_notes="Clinical correlation advised. Mild bilateral apical scarring noted."
    )

    assert isinstance(pdf_bytes, bytes), "PDF generator should return bytes"
    assert len(pdf_bytes) > 1000, f"Generated PDF is suspiciously small ({len(pdf_bytes)} bytes)"
    assert pdf_bytes.startswith(b"%PDF-"), "Generated file does not begin with standard %PDF- magic bytes"
    assert b"%%EOF" in pdf_bytes, "Generated file missing %%EOF trailer"

