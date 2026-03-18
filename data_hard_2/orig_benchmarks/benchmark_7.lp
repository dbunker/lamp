d2(o13).
d2(o4).
d4(o3).
d4(o7).
d3(o16).
d13(o7).
d8(o11).
d10(o3).
d10(o7).
d5(o17).

d9(X) :- not d1(X), d10(X), d4(X).
d12(X) :- d8(X), d1(X), not d3(X).
d9(X) :- d5(X), d3(X), not d4(X).
d8(X) :- not d2(X), d6(X), d13(X).
d7(X) :- not d3(X), d4(X), d13(X).
d8(X) :- d4(X), not d9(X), d11(X).
d11(X) :- d6(X), d1(X), not d8(X).
d4(X) :- not d3(X), d2(X), d13(X).
d10(X) :- d3(X), d4(X), not d5(X).
d1(X) :- d5(X), not d7(X), d12(X).
d8(X) :- d3(X), d2(X), not d6(X).
d11(X) :- d1(X), not d12(X), d6(X).
d3(X) :- d2(X), d4(X), not d8(X).
d6(X) :- d7(X), d8(X), not d1(X).
d6(X) :- d7(X), d9(X), not d3(X).