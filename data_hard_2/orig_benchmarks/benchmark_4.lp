d3(o9).
d5(o4).
d1(o5).
d8(o10).
d0(o7).
d13(o3).
d4(o4).
d1(o10).
d7(o13).
d7(o4).
d6(o3).
d10(o3).
d12(o10).
d2(o3).
d8(o8).

d9(X) :- d5(X), not d10(X), d12(X).
d6(X) :- d2(X), not d5(X), d11(X).
d7(X) :- d3(X), not d9(X), d11(X).
d8(X) :- d0(X), not d7(X), d12(X).
d12(X) :- d7(X), not d2(X), d4(X).
d9(X) :- d8(X), not d3(X), d11(X).
d7(X) :- d9(X), d3(X), not d13(X).
d3(X) :- d10(X), not d8(X), d13(X).
d11(X) :- d8(X), not d6(X), d12(X).
d7(X) :- d3(X), not d10(X), d6(X).
d4(X) :- d6(X), d1(X), not d8(X).
d4(X) :- d7(X), d8(X), not d1(X).
d7(X) :- d8(X), not d6(X), d10(X).
d5(X) :- d0(X), not d11(X), d12(X).
d5(X) :- d3(X), not d4(X), d13(X).