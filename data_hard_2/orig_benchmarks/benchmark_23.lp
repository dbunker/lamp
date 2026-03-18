d14(o14).
d5(o9).
d3(o11).
d4(o15).
d6(o8).
d13(o5).
d2(o16).
d9(o9).
d16(o8).
d0(o14).
d1(o4).
d8(o9).
d11(o2).
d5(o12).
d5(o7).

d1(X) :- d8(X), not d7(X), not d17(X).
d8(X) :- not d2(X), not d14(X), d11(X).
d13(X) :- d8(X), not d16(X), not d17(X).
d10(X) :- not d3(X), not d4(X), d13(X).
d2(X) :- not d5(X), not d13(X), d15(X).
d15(X) :- not d6(X), not d4(X), d13(X).
d1(X) :- d2(X), not d7(X), not d9(X).
d17(X) :- d3(X), not d12(X), not d13(X).
d16(X) :- d14(X), not d6(X), not d3(X).
d17(X) :- not d4(X), not d11(X), d13(X).
d17(X) :- not d3(X), not d16(X), d4(X).
d7(X) :- not d0(X), not d4(X), d16(X).
d12(X) :- not d17(X), d0(X), not d14(X).
d17(X) :- d5(X), not d16(X), not d11(X).
d0(X) :- d5(X), not d4(X), not d16(X).