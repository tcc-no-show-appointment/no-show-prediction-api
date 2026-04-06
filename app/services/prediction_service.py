import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta
from app.services.model_manager import model_manager
from app.constants import PREDICTION_LABEL_SHOW, PREDICTION_LABEL_NO_SHOW
from app.utils.logger import get_logger
from noshow_lib.model_inference import predict as noshow_predict

logger = get_logger(__name__)

async def predict(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict patient no-show using noshow_lib inference pipeline.
    
    Uses per-specialty models loaded at startup. The noshow_lib predict
    function routes each record to the correct specialty model automatically
    based on the specialty_group column (derived during feature engineering).
    """
    logger.info("Starting prediction pipeline")
    
    try:
        config = model_manager.get_config()
        models = model_manager.get_effective_models()
        thresholds = model_manager.get_thresholds()
        logger.info(f"Using {len(models)} effective specialty models")
        
        df = pd.DataFrame([raw_data]) if isinstance(raw_data, dict) else pd.DataFrame(raw_data)
        logger.info(f"Input data shape: {df.shape}, columns: {list(df.columns)}")
        
        result_df = noshow_predict(
            models=models,
            input_data=df,
            config=config,
            thresholds=thresholds,
        )
        
        logger.info(f"Prediction completed. Result shape: {result_df.shape}")
        
        probability_no_show = float(result_df['probability'].iloc[0])
        prediction_value = int(result_df['prediction'].iloc[0])
        probability_show = 1.0 - probability_no_show
        
        result = {
            "prediction": prediction_value,
            "prediction_label": PREDICTION_LABEL_NO_SHOW if prediction_value == 1 else PREDICTION_LABEL_SHOW,
            "probability_show": probability_show,
            "probability_no_show": probability_no_show
        }
        
        # Include specialty_group if available
        if "specialty_group" in result_df.columns:
            result["specialty_group"] = str(result_df['specialty_group'].iloc[0])
        
        logger.info(f"Prediction: {result['prediction_label']} (confidence: {max(probability_show, probability_no_show):.2%})")
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}", exc_info=True)
        raise


async def predict_batch(appointments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Predict patient no-show for multiple appointments in batch.

    Each appointment is predicted in isolation (one-row DataFrame per call) so
    that within-batch cumulative history features computed by noshow_lib's
    feature engineering (e.g. previous_appointments_count, waiting_days_delta)
    are not contaminated by sibling rows in the same request.  This guarantees
    that sending N identical appointments in one batch produces the same result
    as calling /predict N times individually.
    """
    logger.info(f"Starting batch prediction for {len(appointments)} appointments")

    try:
        config = model_manager.get_config()
        models = model_manager.get_effective_models()
        thresholds = model_manager.get_thresholds()
        logger.info(f"Using {len(models)} effective specialty models")

        results = []
        predicted_show_count = 0
        predicted_no_show_count = 0

        for idx, appointment in enumerate(appointments):
            df = pd.DataFrame([appointment])

            result_df = noshow_predict(
                models=models,
                input_data=df,
                config=config,
                output_path=None,
                thresholds=thresholds,
            )

            probability_no_show = float(result_df['probability'].iloc[0])
            prediction_value = int(result_df['prediction'].iloc[0])
            probability_show = 1.0 - probability_no_show

            if prediction_value == 1:
                predicted_no_show_count += 1
            else:
                predicted_show_count += 1

            result = {
                "appointment": appointment,
                "prediction": prediction_value,
                "prediction_label": PREDICTION_LABEL_NO_SHOW if prediction_value == 1 else PREDICTION_LABEL_SHOW,
                "probability_show": probability_show,
                "probability_no_show": probability_no_show,
            }
            if "specialty_group" in result_df.columns:
                result["specialty_group"] = str(result_df['specialty_group'].iloc[0])
            results.append(result)

        logger.info(f"Batch prediction completed. {len(results)} predictions generated.")

        batch_result = {
            "total": len(appointments),
            "predicted_show": predicted_show_count,
            "predicted_no_show": predicted_no_show_count,
            "results": results,
        }

        logger.info(
            f"Batch prediction complete: {predicted_show_count} Show, "
            f"{predicted_no_show_count} No-Show (total: {len(appointments)})"
        )
        return batch_result

    except Exception as e:
        logger.error(f"Error in batch prediction pipeline: {str(e)}", exc_info=True)
        raise


async def predict_range(appointment_data: Dict[str, Any], range_days: int) -> Dict[str, Any]:
    """
    Predict patient no-show across a range of dates for a single appointment.
    """
    logger.info(f"Starting range prediction for {range_days} days")
    
    try:
        original_datetime_str = appointment_data.get('DataHoraConsulta')
        if not original_datetime_str:
            raise ValueError("DataHoraConsulta is required for range prediction")
        
        try:
            original_datetime = datetime.fromisoformat(original_datetime_str.replace('Z', '+00:00'))
        except:
            for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                try:
                    original_datetime = datetime.strptime(original_datetime_str, fmt)
                    break
                except:
                    continue
            else:
                raise ValueError(f"Could not parse date format: {original_datetime_str}")
        
        original_date = original_datetime.date()
        original_time = original_datetime.time()
        
        logger.info(f"Original appointment: {original_date} at {original_time}")
        
        appointments_variations = []
        date_mapping = []
        
        for day_offset in range(range_days):
            target_date = original_date + timedelta(days=day_offset)
            target_datetime = datetime.combine(target_date, original_time)
            
            appointment_variation = appointment_data.copy()
            appointment_variation['DataHoraConsulta'] = target_datetime.strftime('%Y-%m-%dT%H:%M:%S')
            
            appointments_variations.append(appointment_variation)
            date_mapping.append(target_date.strftime('%Y-%m-%d'))
        
        logger.info(f"Generated {len(appointments_variations)} appointment variations")
        
        config = model_manager.get_config()
        models = model_manager.get_effective_models()
        thresholds = model_manager.get_thresholds()
        
        df = pd.DataFrame(appointments_variations)
        
        logger.info("Running range inference using noshow_lib.model_inference.predict")
        result_df = noshow_predict(
            models=models,
            input_data=df,
            config=config,
            output_path=None,
            thresholds=thresholds,
        )
        
        logger.info(f"Range prediction completed. Result shape: {result_df.shape}")
        
        predictions = []
        probabilities_no_show = []
        
        for idx in range(len(result_df)):
            probability_no_show = float(result_df['probability'].iloc[idx])
            prediction_value = int(result_df['prediction'].iloc[idx])
            probability_show = 1.0 - probability_no_show
            
            probabilities_no_show.append(probability_no_show)
            
            date_pred = {
                'date': date_mapping[idx],
                'prediction': prediction_value,
                'prediction_label': PREDICTION_LABEL_NO_SHOW if prediction_value == 1 else PREDICTION_LABEL_SHOW,
                'probability_no_show': probability_no_show,
                'probability_show': probability_show
            }
            predictions.append(date_pred)
        
        avg_prob = sum(probabilities_no_show) / len(probabilities_no_show)
        min_prob = min(probabilities_no_show)
        max_prob = max(probabilities_no_show)
        
        best_idx = probabilities_no_show.index(min_prob)
        worst_idx = probabilities_no_show.index(max_prob)
        
        summary = {
            'avg_probability_no_show': round(avg_prob, 4),
            'min_probability_no_show': round(min_prob, 4),
            'max_probability_no_show': round(max_prob, 4),
            'best_date': date_mapping[best_idx],
            'worst_date': date_mapping[worst_idx]
        }
        
        range_result = {
            'original_appointment_date': original_date.strftime('%Y-%m-%d'),
            'original_appointment_time': original_time.strftime('%H:%M:%S'),
            'patient_id': appointment_data.get('idUnicoPaciente'),
            'predictions': predictions,
            'range_days': range_days,
            'summary': summary
        }
        
        logger.info(
            f"Range prediction complete: avg no-show risk {avg_prob:.2%}, "
            f"best date: {summary['best_date']} ({min_prob:.2%}), "
            f"worst date: {summary['worst_date']} ({max_prob:.2%})"
        )
        
        return range_result
        
    except Exception as e:
        logger.error(f"Error in range prediction pipeline: {str(e)}", exc_info=True)
        raise
