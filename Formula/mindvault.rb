class Mindvault < Formula
  include Language::Python::Virtualenv

  desc "Local-first second brain RAG system"
  homepage "https://github.com/calebthecm/MindVault"
  url "https://github.com/calebthecm/MindVault/archive/refs/tags/v0.5.3.tar.gz"
  sha256 "e3cb2a5ca7a079fa27f55c0453a9c335df832475d5089fcb320c923b5d44306"
  license "Proprietary"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    system "#{bin}/mindvault", "--help"
  end
end