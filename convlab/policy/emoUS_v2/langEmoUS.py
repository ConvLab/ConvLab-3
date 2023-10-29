import torch
from convlab.policy.emoUS_v2.semanticEmoUS import UserActionPolicy as semanticEmoUS


class UserActionPolicy(semanticEmoUS):
    def __init__(self, model_checkpoint, mode="language", max_turn=40, **kwargs):
        super().__init__(model_checkpoint, mode, max_turn, **kwargs)

    def _generate_action(self, raw_inputs, sys_act=None, mode="max", allow_general_intent=True, emotion_mode="normal", emotion=None, golden_action=None):
        self.kg.parse_input(raw_inputs, sys_act)
        model_input = self.vector.encode(
            raw_inputs, self.max_in_len, do_padding=self.padding)
        # start token
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = 0
        if self.model.model_type == "encoder_decoder":
            pos = self._update_seq([0], 0)
        # else:
        #     pos = self._update_seq([1], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)

        if self.use_sentiment and self.emotion_mid:
            pos = self._sent_act_emo(
                pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=emotion, golden_action=golden_action)
        elif self.use_sentiment and not self.emotion_mid:
            pos = self._sent_emo_act(
                pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=emotion, golden_action=golden_action)
        elif not self.use_sentiment and self.emotion_mid:
            pos = self._act_emo(
                pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=emotion, golden_action=golden_action)
        else:  # defalut method
            pos = self._emo_act(
                pos, model_input, mode, emotion_mode, allow_general_intent, golden_emotion=emotion, golden_action=golden_action)

        if self.only_action:
            # return semantic action. Don't need to generate text
            return self.vector.decode(self.seq[0, :pos])

        pos = self._update_seq(self.token_map.get_id("start_text"), pos)
        text = self._get_text(model_input, pos)

        return text
