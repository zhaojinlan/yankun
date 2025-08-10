
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


DEFAULT_CONFIG_LIST = [{
    "model": "deepseek-v3-250324",
    "api_key": "2bac91ba-9d5e-403f-8e0a-30a5fce5bde6",
    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
}]



def linguist_psychologist_discussion(topic: str, max_turns: int = 4):
    """语言学家与心理学家多智能体讨论。

    Args:
        topic: 讨论主题
        max_turns: 最多轮次

    Returns:
        对话历史列表
    """
    linguist = AssistantAgent(
        name="linguist",
        system_message="你是一名语言学家，删除",
        llm_config={"config_list": DEFAULT_CONFIG_LIST},
    )
    psychologist = AssistantAgent(
        name="psychologist",
        system_message="你是一名心理学家，擅长从认知与情绪角度分析问题，并提供心理学视角的见解。请用中文回答。",
        llm_config={"config_list": DEFAULT_CONFIG_LIST},
    )

    groupchat = GroupChat(
        agents=[linguist, psychologist],
        speaker_selection_method="round_robin",
        allow_repeat_speaker=False,
        messages=[],
    )
    manager = GroupChatManager(
        name="manager",
        groupchat=groupchat,
        llm_config={"config_list": DEFAULT_CONFIG_LIST},
    )
    requester = UserProxyAgent(
        name="requester",
        human_input_mode="NEVER",
        code_execution_config=False,
    )
    conversation = requester.initiate_chat(manager, message=topic, max_turns=max_turns)

    # 生成简短结论
    summary_agent = AssistantAgent(
        name="summarizer",
        system_message="你是一名会议记录员，擅长总结，请基于先前对话，用不超过100字的中文总结讨论并给出明确结论。",
        llm_config={"config_list": DEFAULT_CONFIG_LIST},
    )

    history_text = "\n".join([
        f"{m.get('name', m.get('role', ''))}: {m.get('content', '')}"
        for m in conversation.chat_history
    ])
    requester.initiate_chat(summary_agent, message=history_text, max_turns=1)
    summary = summary_agent.chat_history[-1]["content"]  # type: ignore[index]

    return conversation.chat_history, summary  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 以下为演示入口，真正业务环境可直接 import 上述函数调用
# ---------------------------------------------------------------------------

def main():

 

    # 语言学家与心理学家讨论演示
    discussion_history, conclusion = linguist_psychologist_discussion("你好，您的母亲现在病危我很理解你的心情", max_turns=2)
    print("\n[语言学家与心理学家讨论记录]")
    for msg in discussion_history:
        print(f"{msg.get('name', msg.get('role', ''))}: {msg.get('content', '')}")
    print("\n[总结]\n" + conclusion)


if __name__ == "__main__":
    main() 