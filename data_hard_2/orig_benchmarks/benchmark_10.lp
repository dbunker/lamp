d10(o9).
d3(o13).
d2(o11).
d13(o1).
d12(o4).
d9(o7).
d13(o5).
d2(o6).
d12(o16).
d5(o1).
d6(o4).
d5(o5).
d0(o6).
d7(o16).
d1(o16).

d7(X) :- d4(X), d12(X), not d8(X).
d4(X) :- d0(X), d2(X), not d6(X).
d0(X) :- not d6(X), d2(X), d13(X).
d1(X) :- d0(X), d3(X), not d5(X).
d9(X) :- d6(X), not d4(X), d11(X).
d10(X) :- d5(X), not d8(X), d13(X).
d4(X) :- d8(X), d7(X), not d9(X).
d8(X) :- d9(X), d12(X), not d5(X).
d11(X) :- d2(X), not d5(X), d13(X).
d5(X) :- d6(X), not d10(X), d11(X).
d1(X) :- d10(X), not d7(X), d13(X).
d13(X) :- d7(X), not d12(X), d11(X).
d2(X) :- d3(X), not d13(X), d11(X).
d2(X) :- d7(X), d4(X), not d11(X).
d6(X) :- d2(X), not d7(X), d13(X).