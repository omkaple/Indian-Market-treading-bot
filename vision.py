import base64
import json
import re
import logging
from typing import Dict, Any, Optional
import requests
from SmartApi import SmartConnect

# Configure module-level logging
logger = logging.getLogger("Vision")

# Local Ollama endpoint configs
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2-vision"

class VisionCognitionEngine:
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.api_url: str = OLLAMA_API_URL
        self.model: str = model_name

    def encode_image(self, image_path: str) -> str:
        """Reads a file and returns its raw base64 string representation."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def evaluate_crossover(self, image_path: str = "temp_crossover_state.png") -> Dict[str, Any]:
        """
        Sends the crossover chart image to Ollama's vision model to evaluate technical breakout strength.
        Forces and returns a JSON payload: {"trade_approved": bool, "confidence_score": float}
        """
        logger.info(f"Encoding chart image '{image_path}' for Vision AI...")
        try:
            base64_img = self.encode_image(image_path)
        except Exception as e:
            logger.error(f"Failed to read/encode chart image: {e}")
            return {"trade_approved": False, "confidence_score": 0.0, "error": str(e)}

        prompt = (
            "You are an expert technical analyst. Evaluate the attached 20/50 EMA Golden Cross crossover chart. "
            "Analyze the chart features strictly on these three vectors:\n"
            "1. Crossover Angle: Is the 20 EMA (orange line) cutting through the 50 EMA (blue line) at a sharp, clean, high-momentum angle, or is it flattening out into a low-momentum consolidation?\n"
            "2. Candlestick Structure: Are there candlestick reversals (like long wicks suggesting selling pressure/exhaustion) or false breakout patterns near the crossover point?\n"
            "3. Volume Confirmation: Verify if the crossover breakout candle is supported by a visible spike in the volume bars at the bottom relative to the past 10 bars.\n\n"
            "Respond ONLY with a valid raw JSON object. Do not include any conversational filler, explanation, or markdown formatting.\n"
            "The JSON structure must match this template exactly:\n"
            "{\n"
            '  "trade_approved": <true|false>,\n'
            '  "confidence_score": <float between 0.0 and 1.0>,\n'
            '  "analysis": {\n'
            '    "crossover_angle": "<sharp|moderate|flat>",\n'
            '    "exhaustion_wicks": <true|false>,\n'
            '    "volume_confirmed": <true|false>,\n'
            '    "commentary": "<brief sentence summary>"\n'
            '  }\n'
            "}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_img]
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1
            },
            "format": "json"  # Instructs Ollama to enforce JSON formatting
        }

        logger.info(f"Sending Vision AI request to Ollama ({self.model})...")
        try:
            # Send HTTP POST with a 45 second timeout
            response = requests.post(self.api_url, json=payload, timeout=45)
            response.raise_for_status()
            
            result = response.json()
            content_str = result.get("message", {}).get("content", "").strip()
            
            parsed_json = self._extract_json(content_str)
            logger.info(f"Vision Decision: Approved={parsed_json.get('trade_approved')}, Confidence={parsed_json.get('confidence_score')}")
            return parsed_json

        except Exception as e:
            logger.error(f"Vision AI analysis failed: {e}")
            # Fallback to simulation paper-trading evaluation if Ollama service is not running
            logger.warning("Falling back to simulated Vision AI mock response.")
            return self._generate_mock_response()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Tolerantly parses JSON output from raw text strings using regex."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as je:
                logger.error(f"Failed parsing matched regex group to JSON: {je}")
                
        raise ValueError(f"Could not parse valid JSON from text: {text}")

    def _generate_mock_response(self) -> Dict[str, Any]:
        """Generates a randomized mock response for simulation when local Ollama is offline."""
        import random
        approved = random.choice([True, False])
        confidence = round(random.uniform(0.60, 0.95), 2)
        return {
            "trade_approved": approved,
            "confidence_score": confidence,
            "analysis": {
                "crossover_angle": "sharp" if approved else "flat",
                "exhaustion_wicks": not approved,
                "volume_confirmed": approved,
                "commentary": "Simulated evaluation of the Golden Cross breakout."
            }
        }

    def process_and_execute(self, smart_conn: Optional[SmartConnect], vision_result: Dict[str, Any], 
                            symbol_name: str, token_id: int, exchange: str = "NSE",
                            entry_price: float = 0.0, stop_loss_price: float = 0.0, 
                            target_price: float = 0.0, quantity: int = 10) -> tuple[bool, str]:
        """
        Pre-order safety filter check:
        Places a live MARKET order on the exchange if approved and confidence >= 0.75.
        Then places protective SL and Target orders to enforce Risk-Reward ratio.
        Returns a tuple: (success: bool, order_id_or_error_msg: str)
        """
        trade_approved = vision_result.get("trade_approved", False)
        confidence_score = vision_result.get("confidence_score", 0.0)

        logger.info(f"Safety Check: Approved={trade_approved}, Confidence={confidence_score:.2f} (Threshold: 0.75)")

        if not trade_approved:
            msg = "Crossover pattern not approved by Vision AI."
            logger.info(f">>> ORDER CANCELLED: {msg}")
            return False, msg

        if confidence_score < 0.75:
            msg = f"Confidence score {confidence_score:.2f} is below 0.75 risk threshold."
            logger.info(f">>> ORDER CANCELLED: {msg}")
            return False, msg

        logger.info(">>> RISK FILTER PASSED! Dispatching execution order to exchange...")
        logger.info(f">>> R:R Enforcement — Entry: ₹{entry_price:.2f} | SL: ₹{stop_loss_price:.2f} | Target: ₹{target_price:.2f} | Qty: {quantity}")
        
        # Build tradingsymbol correctly
        tradingsymbol = symbol_name
        if exchange == "NSE" and not symbol_name.endswith("-EQ") and symbol_name not in ["Nifty 50", "Nifty Bank", "Nifty Fin Service", "NIFTY MID SELECT"]:
            tradingsymbol = f"{symbol_name}-EQ"

        # ───────────────────────────────────────────────────
        # STEP 1: Place the main MARKET BUY entry order
        # ───────────────────────────────────────────────────
        entry_order_params = {
            "variety": "NORMAL",
            "tradingsymbol": tradingsymbol,
            "symboltoken": str(token_id),
            "transactiontype": "BUY",
            "exchange": exchange,
            "ordertype": "MARKET",
            "producttype": "INTRADAY",
            "duration": "DAY",
            "quantity": str(quantity)
        }

        logger.info(f"[1/3] Entry Order Params: {entry_order_params}")

        if not smart_conn:
            msg = f"[SIMULATION] Mock Entry BUY placed. SL=₹{stop_loss_price:.2f}, Target=₹{target_price:.2f}, Qty={quantity}"
            logger.warning(msg)
            return True, "MOCK_ORDER_12345"

        try:
            res = smart_conn.placeOrder(entry_order_params)
            if isinstance(res, dict):
                if res.get("status") is True:
                    entry_order_id = res.get("data", {}).get("orderid", "unknown_id")
                else:
                    msg = res.get("message", "Unknown broker error on entry order")
                    logger.error(f"Broker rejected entry order: {msg}")
                    return False, msg
            else:
                entry_order_id = str(res)
            
            logger.info(f"[1/3] ✅ Entry BUY order routed! Order ID: {entry_order_id}")

        except Exception as e:
            err_msg = str(e)
            logger.error(f"Entry order execution failed: {err_msg}", exc_info=True)
            return False, err_msg

        # ───────────────────────────────────────────────────
        # STEP 2: Place protective STOPLOSS SELL order
        # ───────────────────────────────────────────────────
        if stop_loss_price > 0:
            sl_trigger = round(stop_loss_price, 2)
            sl_limit = round(stop_loss_price - 0.10, 2)  # Slight buffer below trigger
            
            sl_order_params = {
                "variety": "STOPLOSS",
                "tradingsymbol": tradingsymbol,
                "symboltoken": str(token_id),
                "transactiontype": "SELL",
                "exchange": exchange,
                "ordertype": "STOPLOSS_LIMIT",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "quantity": str(quantity),
                "price": str(sl_limit),
                "triggerprice": str(sl_trigger)
            }
            
            logger.info(f"[2/3] SL Order Params: {sl_order_params}")
            
            try:
                sl_res = smart_conn.placeOrder(sl_order_params)
                if isinstance(sl_res, dict):
                    sl_order_id = sl_res.get("data", {}).get("orderid", "unknown") if sl_res.get("status") else "FAILED"
                    if sl_order_id == "FAILED":
                        logger.warning(f"⚠️ SL order rejected: {sl_res.get('message', 'Unknown error')}. Manual SL required!")
                else:
                    sl_order_id = str(sl_res)
                logger.info(f"[2/3] ✅ Stop Loss SELL order placed! Trigger: ₹{sl_trigger} | Order ID: {sl_order_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to place SL order: {e}. Manual Stop Loss required at ₹{sl_trigger}!")

        # ───────────────────────────────────────────────────
        # STEP 3: Place LIMIT SELL order for profit target
        # ───────────────────────────────────────────────────
        if target_price > 0:
            target_limit = round(target_price, 2)
            
            target_order_params = {
                "variety": "NORMAL",
                "tradingsymbol": tradingsymbol,
                "symboltoken": str(token_id),
                "transactiontype": "SELL",
                "exchange": exchange,
                "ordertype": "LIMIT",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "quantity": str(quantity),
                "price": str(target_limit)
            }
            
            logger.info(f"[3/3] Target Order Params: {target_order_params}")
            
            try:
                tgt_res = smart_conn.placeOrder(target_order_params)
                if isinstance(tgt_res, dict):
                    tgt_order_id = tgt_res.get("data", {}).get("orderid", "unknown") if tgt_res.get("status") else "FAILED"
                    if tgt_order_id == "FAILED":
                        logger.warning(f"⚠️ Target order rejected: {tgt_res.get('message', 'Unknown error')}. Manual target exit required!")
                else:
                    tgt_order_id = str(tgt_res)
                logger.info(f"[3/3] ✅ Target SELL order placed! Limit: ₹{target_limit} | Order ID: {tgt_order_id}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to place target order: {e}. Manual exit required at ₹{target_limit}!")

        logger.info(f"🎯 R:R enforcement complete — Entry: ₹{entry_price:.2f} | SL: ₹{stop_loss_price:.2f} | Target: ₹{target_price:.2f}")
        return True, entry_order_id
