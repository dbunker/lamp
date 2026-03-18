d13(o12).
d8(o13).
d9(o4).
d1(o7).
d12(o6).
d12(o9).
d4(o4).
d2(o6).
d11(o5).
d3(o0).
d12(o8).
d6(o11).
d0(o9).
d11(o2).
d1(o9).

d4(X) :- not d1(X), d5(X).
d3(X) :- not d4(X), d12(X).
d5(X) :- d4(X), not d12(X).
d4(X) :- d1(X), not d11(X).
d6(X) :- not d4(X), d7(X).
d0(X) :- not d12(X), d7(X).
d2(X) :- not d8(X), d1(X).
d3(X) :- not d12(X), d7(X).
d7(X) :- not d1(X), d8(X).
d4(X) :- d8(X), not d2(X).
d11(X) :- not d7(X), d10(X).
d8(X) :- d6(X), not d2(X).
d7(X) :- d2(X), not d12(X).
d7(X) :- d10(X), not d6(X).
d10(X) :- not d4(X), d7(X).
d10(X) :- not d3(X), d6(X).
d7(X) :- not d0(X), d6(X).
d4(X) :- d12(X), not d6(X).
d12(X) :- not d7(X), d6(X).
d11(X) :- not d0(X), d6(X).