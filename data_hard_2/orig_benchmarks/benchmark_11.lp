d11(o6).
d4(o11).
d5(o4).
d11(o8).
d5(o8).
d1(o10).
d4(o4).
d1(o15).
d5(o1).
d8(o3).
d0(o9).
d13(o4).
d10(o12).
d5(o0).
d5(o7).

d12(X) :- not d0(X), d3(X), not d4(X).
d2(X) :- d7(X), not d3(X), not d11(X).
d9(X) :- d1(X), not d4(X), not d12(X).
d4(X) :- d3(X), not d7(X), not d9(X).
d7(X) :- not d1(X), not d9(X), d11(X).
d4(X) :- not d10(X), not d9(X), d13(X).
d4(X) :- d7(X), not d12(X), not d13(X).
d13(X) :- not d2(X), d6(X), not d8(X).
d6(X) :- not d5(X), not d8(X), d13(X).
d12(X) :- d3(X), not d13(X), not d8(X).
d1(X) :- not d4(X), d10(X), not d5(X).
d13(X) :- d6(X), not d4(X), not d8(X).
d9(X) :- d8(X), not d2(X), not d5(X).
d7(X) :- d8(X), not d6(X), not d10(X).
d3(X) :- not d2(X), d6(X), not d12(X).