d13(o8).
d9(o7).
d7(o0).
d3(o6).
d1(o3).
d1(o13).
d8(o11).
d12(o3).
d1(o6).
d11(o4).

d10(X) :- d9(X), d2(X), not d1(X).
d5(X) :- d7(X), not d6(X), d12(X).
d3(X) :- d9(X), d8(X), not d7(X).
d6(X) :- d3(X), d1(X), not d12(X).
d0(X) :- not d3(X), d1(X), d4(X).
d10(X) :- d13(X), not d7(X), d11(X).
d1(X) :- d0(X), d4(X), not d11(X).
d7(X) :- d5(X), not d4(X), d11(X).
d11(X) :- d2(X), not d4(X), d12(X).
d6(X) :- d7(X), d10(X), not d3(X).
d4(X) :- d9(X), not d2(X), d3(X).
d2(X) :- not d6(X), d1(X), d12(X).
d2(X) :- d3(X), d12(X), not d9(X).
d5(X) :- d1(X), d12(X), not d8(X).
d3(X) :- d8(X), d12(X), not d0(X).