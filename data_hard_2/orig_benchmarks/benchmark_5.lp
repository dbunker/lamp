d11(o7).
d0(o3).
d11(o12).
d7(o13).
d0(o2).
d4(o1).
d3(o7).
d3(o11).
d2(o11).
d10(o8).
d10(o11).
d4(o9).
d8(o3).
d7(o10).
d2(o4).

d2(X) :- not d5(X), not d10(X), d3(X).
d7(X) :- not d5(X), not d12(X), d11(X).
d13(X) :- not d9(X), not d11(X), d3(X).
d7(X) :- d1(X), not d11(X), not d2(X).
d5(X) :- not d4(X), not d13(X), d12(X).
d2(X) :- d0(X), not d9(X), not d10(X).
d9(X) :- not d4(X), not d7(X), d3(X).
d4(X) :- not d8(X), d1(X), not d3(X).
d0(X) :- d8(X), not d11(X), not d12(X).
d4(X) :- d2(X), not d0(X), not d6(X).
d4(X) :- not d0(X), d11(X), not d2(X).
d12(X) :- not d8(X), d10(X), not d2(X).
d7(X) :- not d5(X), not d9(X), d6(X).
d8(X) :- not d13(X), d7(X), not d11(X).
d4(X) :- not d5(X), not d13(X), d11(X).
d6(X) :- not d7(X), d1(X), not d13(X).
d1(X) :- not d7(X), d9(X), not d10(X).
d6(X) :- not d1(X), not d7(X), d12(X).
d4(X) :- d2(X), not d6(X), not d3(X).
d9(X) :- not d8(X), d10(X), not d6(X).