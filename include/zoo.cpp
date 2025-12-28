#pragma once

/**
 * @file zoo.h
 * @brief Main include file for the xInfer Model Zoo.
 *
 * This header provides access to all high-level, task-oriented application
 * modules in the xInfer ecosystem.
 */

// --- Accessibility ---
#include <xinfer/zoo/accessibility/sign_translator.h>
#include <xinfer/zoo/accessibility/speech_augmenter.h>
#include <xinfer/zoo/accessibility/visual_assistant.h>

// --- AEC (Architecture, Engineering, Construction) ---
#include <xinfer/zoo/aec/blueprint_auditor.h>
#include <xinfer/zoo/aec/site_safety_monitor.h>

// --- Audio ---
#include <xinfer/zoo/audio/classifier.h>
#include <xinfer/zoo/audio/event_detector.h>
#include <xinfer/zoo/audio/language_identifier.h>
#include <xinfer/zoo/audio/music_source_separator.h>
#include <xinfer/zoo/audio/speaker_identifier.h>
#include <xinfer/zoo/audio/speech_recognizer.h>

// --- Chemistry ---
#include <xinfer/zoo/chemistry/molecule_analyzer.h>
#include <xinfer/zoo/chemistry/reaction_forecaster.h>

// --- Civil Engineering ---
#include <xinfer/zoo/civil/grid_inspector.h>
#include <xinfer/zoo/civil/pavement_analyzer.h>
#include <xinfer/zoo/civil/structural_inspector.h>

// --- Cybersecurity ---
#include <xinfer/zoo/cybersecurity/malware_classifier.h>
#include <xinfer/zoo/cybersecurity/network_detector.h>

// --- Document AI ---
#include <xinfer/zoo/document/handwriting_recognizer.h>
#include <xinfer/zoo/document/layout_parser.h>
#include <xinfer/zoo/document/signature_detector.h>
#include <xinfer/zoo/document/table_extractor.h>

// --- Drones ---
#include <xinfer/zoo/drones/navigation_policy.h>

// --- DSP ---
#include <xinfer/zoo/dsp/signal_filter.h>
#include <xinfer/zoo/dsp/spectrogram.h>

// --- Education ---
#include <xinfer/zoo/education/grader.h>
#include <xinfer/zoo/education/presentation_coach.h>
#include <xinfer/zoo/education/tutor.h>

// --- Energy ---
#include <xinfer/zoo/energy/seismic_interpreter.h>
#include <xinfer/zoo/energy/turbine_inspector.h>
#include <xinfer/zoo/energy/well_log_analyzer.h>

// --- Fashion ---
#include <xinfer/zoo/fashion/fabric_inspector.h>
#include <xinfer/zoo/fashion/pattern_generator.h>
#include <xinfer/zoo/fashion/trend_forecaster.h>
#include <xinfer/zoo/fashion/virtual_tryon.h>

// --- Gaming ---
#include <xinfer/zoo/gaming/npc_behavior_policy.h>

// --- Generative ---
#include <xinfer/zoo/generative/colorizer.h>
#include <xinfer/zoo/generative/dcgan.h>
#include <xinfer/zoo/generative/diffusion_pipeline.h>
#include <xinfer/zoo/generative/image_to_video.h>
#include <xinfer/zoo/generative/inpainter.h>
#include <xinfer/zoo/generative/outpainter.h>
#include <xinfer/zoo/generative/style_transfer.h>
#include <xinfer/zoo/generative/super_resolution.h>
#include <xinfer/zoo/generative/text_to_3d.h>
#include <xinfer/zoo/generative/text_to_speech.h>
#include <xinfer/zoo/generative/video_frame_interpolation.h>
#include <xinfer/zoo/generative/voice_converter.h>

// --- Geospatial ---
#include <xinfer/zoo/geospatial/building_segmenter.h>
#include <xinfer/zoo/geospatial/crop_monitor.h>
#include <xinfer/zoo/geospatial/disaster_assessor.h>
#include <xinfer/zoo/geospatial/maritime_detector.h>
#include <xinfer/zoo/geospatial/road_extractor.h>

// --- HCI (Human-Computer Interaction) ---
#include <xinfer/zoo/hci/emotion_recognizer.h>
#include <xinfer/zoo/hci/gaze_tracker.h>
#include <xinfer/zoo/hci/lip_reader.h>

// --- HFT (High-Frequency Trading) ---
#include <xinfer/zoo/hft/order_execution_policy.h>

// --- Insurance ---
#include <xinfer/zoo/insurance/claim_assessor.h>
#include <xinfer/zoo/insurance/document_analyzer.h>
#include <xinfer/zoo/insurance/property_assessor.h>

// --- Legal ---
#include <xinfer/zoo/legal/contract_analyzer.h>
#include <xinfer/zoo/legal/ediscovery_classifier.h>

// --- Live Events ---
#include <xinfer/zoo/live_events/broadcast_director.h>
#include <xinfer/zoo/live_events/crowd_analyzer.h>
#include <xinfer/zoo/live_events/replay_generator.h>
#include <xinfer/zoo/live_events/sports_analyzer.h>

// --- Logistics ---
#include <xinfer/zoo/logistics/damage_assessor.h>
#include <xinfer/zoo/logistics/fill_estimator.h>
#include <xinfer/zoo/logistics/inventory_scanner.h>

// --- Maritime ---
#include <xinfer/zoo/maritime/collision_avoidance.h>
#include <xinfer/zoo/maritime/docking_system.h>
#include <xinfer/zoo/maritime/port_analyzer.h>

// --- Media Forensics ---
#include <xinfer/zoo/media_forensics/audio_authenticator.h>
#include <xinfer/zoo/media_forensics/deepfake_detector.h>
#include <xinfer/zoo/media_forensics/provenance_tracker.h>

// --- Medical ---
#include <xinfer/zoo/medical/artery_analyzer.h>
#include <xinfer/zoo/medical/cell_segmenter.h>
#include <xinfer/zoo/medical/pathology_assistant.h>
#include <xinfer/zoo/medical/retina_scanner.h>
#include <xinfer/zoo/medical/tumor_detector.h>
#include <xinfer/zoo/medical/ultrasound_guide.h>

// --- NLP ---
#include <xinfer/zoo/nlp/classifier.h>
#include <xinfer/zoo/nlp/code_generator.h>
#include <xinfer/zoo/nlp/embedder.h>
#include <xinfer/zoo/nlp/keyword_extractor.h>
#include <xinfer/zoo/nlp/ner.h>
#include <xinfer/zoo/nlp/question_answering.h>
#include <xinfer/zoo/nlp/summarizer.h>
#include <xinfer/zoo/nlp/text_generator.h>
#include <xinfer/zoo/nlp/translator.h>

// --- Recruitment ---
#include <xinfer/zoo/recruitment/candidate_matcher.h>
#include <xinfer/zoo/recruitment/resume_parser.h>

// --- Recycling ---
#include <xinfer/zoo/recycling/landfill_monitor.h>
#include <xinfer/zoo/recycling/sorter.h>

// --- Retail ---
#include <xinfer/zoo/retail/customer_analyzer.h>
#include <xinfer/zoo/retail/demand_forecaster.h>
#include <xinfer/zoo/retail/shelf_auditor.h>
#include <xinfer/zoo/retail/smart_checkout.h>

// --- Robotics ---
#include <xinfer/zoo/robotics/assembly_policy.h>
#include <xinfer/zoo/robotics/grasp_planner.h>
#include <xinfer/zoo/robotics/soft_body_simulator.h>
#include <xinfer/zoo/robotics/visual_servo.h>

// --- Science ---
#include <xinfer/zoo/science/climate_simulator.h>
#include <xinfer/zoo/science/particle_tracker.h>
#include <xinfer/zoo/science/transient_detector.h>

// --- Space ---
#include <xinfer/zoo/space/data_triage_engine.h>
#include <xinfer/zoo/space/debris_tracker.h>
#include <xinfer/zoo/space/docking_controller.h>

// --- Special ---
#include <xinfer/zoo/special/genomics.h>
#include <xinfer/zoo/special/hft.h>
#include <xinfer/zoo/special/physics.h>

// --- Telecom ---
#include <xinfer/zoo/telecom/network_control_policy.h>

// --- 3D ---
#include <xinfer/zoo/threed/pointcloud_detector.h>
#include <xinfer/zoo/threed/pointcloud_segmenter.h>
#include <xinfer/zoo/threed/reconstructor.h>
#include <xinfer/zoo/threed/slam_accelerator.h>

// --- Time Series ---
#include <xinfer/zoo/timeseries/anomaly_detector.h>
#include <xinfer/zoo/timeseries/classifier.h>
#include <xinfer/zoo/timeseries/forecaster.h>

// --- Vision ---
#include <xinfer/zoo/vision/animal_tracker.h>
#include <xinfer/zoo/vision/anomaly_detector.h>
#include <xinfer/zoo/vision/barcode_scanner.h>
#include <xinfer/zoo/vision/change_detector.h>
#include <xinfer/zoo/vision/classifier.h>
#include <xinfer/zoo/vision/depth_estimator.h>
#include <xinfer/zoo/vision/detector.h>
#include <xinfer/zoo/vision/face_detector.h>
#include <xinfer/zoo/vision/face_recognizer.h>
#include <xinfer/zoo/vision/hand_tracker.h>
#include <xinfer/zoo/vision/image_deblur.h>
#include <xinfer/zoo/vision/image_similarity.h>
#include <xinfer/zoo/vision/instance_segmenter.h>
#include <xinfer/zoo/vision/license_plate_recognizer.h>
#include <xinfer/zoo/vision/low_light_enhancer.h>
#include <xinfer/zoo/vision/ocr.h>
#include <xinfer/zoo/vision/optical_flow.h>
#include <xinfer/zoo/vision/pose_estimator.h>
#include <xinfer/zoo/vision/segmenter.h>
#include <xinfer/zoo/vision/smoke_flame_detector.h>
#include <xinfer/zoo/vision/vehicle_identifier.h>