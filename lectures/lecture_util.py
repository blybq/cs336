from edtrace import link


def named_link(url: str, name: str) -> str:
    return link(title=f" [{name}]", url=url)


def article_link(url: str) -> str:
    return link(title=" [article]", url=url)


def post_link(url: str) -> str:
    return link(title=" [post]", url=url)


def video_link(url: str) -> str:
    return link(title=" [video]", url=url)


def get_local_url(path: str) -> str:
    return "https://github.com/stanford-cs336/lectures/blob/spring2026/" + path
