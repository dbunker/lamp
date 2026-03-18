d0(o17).
d10(o2).
d6(o16).
d12(o12).
d9(o14).
d6(o6).
d4(o5).
d0(o9).
d15(o12).
d11(o2).
d16(o1).
d1(o6).
d14(o6).
d13(o6).
d12(o5).

d16(X) :- d10(X), not d15(X).
d0(X) :- not d1(X), d13(X).
d12(X) :- not d8(X), d16(X).
d2(X) :- not d7(X), d14(X).
d8(X) :- d16(X), not d14(X).
d11(X) :- not d8(X), d15(X).
d0(X) :- not d12(X), d13(X).
d3(X) :- not d0(X), d4(X).
d14(X) :- not d4(X), d7(X).
d5(X) :- d8(X), not d11(X).
d11(X) :- not d10(X), d9(X).
d17(X) :- d4(X), not d12(X).
d10(X) :- d12(X), not d6(X).
d17(X) :- d7(X), not d15(X).
d16(X) :- d4(X), not d3(X).
d3(X) :- not d4(X), d14(X).
d7(X) :- not d11(X), d14(X).
d14(X) :- not d4(X), d13(X).
d2(X) :- not d1(X), d8(X).
d3(X) :- not d1(X), d17(X).